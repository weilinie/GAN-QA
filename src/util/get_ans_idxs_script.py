import lutorpy as lua
import numpy as np

# import lua module
tokenize = lua.eval(
    '''
function(line, opt)

  if opt.mode == 'space' then
    local index = 1
    local tokens = {}
    while index <= line:len() do
      local sepStart, sepEnd = line:find(' ', index)
      local sub
      if not sepStart then
        sub = line:sub(index)
        table.insert(tokens, sub)
        break
      else
        sub = line:sub(index, sepStart - 1)
        table.insert(tokens, sub)
        index = sepEnd + 1
      end
    end

    return tokens
  end

  local tokens = {}
  -- contains the current token
  local curtok = ''
  -- keep category of the previous character
  local space = true
  local letter = false
  local prev_alphabet
  local number = false
  local other = false
  local placeholder = false

  -- iterate on utf-8 characters
  for v, c, nextv in unicode.utf8_iter(line) do
    if placeholder then
      if c == separators.ph_marker_close then
        curtok = curtok .. c
        letter = true
        prev_alphabet = 'placeholder'
        placeholder = false
        space = false
      else
        if unicode.isSeparator(v) then
          c = string.format(separators.protected_character.."%04x", v)
        end
        curtok = curtok .. c
      end
    elseif c == separators.ph_marker_open then
      if space == false then
        if opt.joiner_annotate and not(opt.joiner_new) then
          curtok = curtok .. opt.joiner
        end
        table.insert(tokens, curtok)
        if opt.joiner_annotate and opt.joiner_new then
          table.insert(tokens, opt.joiner)
        end
      elseif other == true then
        if opt.joiner_annotate then
          if curtok == '' then
            if opt.joiner_new then table.insert(tokens, opt.joiner)
            else tokens[#tokens] = tokens[#tokens] .. opt.joiner end
          end
        end
      end
      curtok = c
      placeholder = true
    elseif unicode.isSeparator(v) then
      if space == false then
        table.insert(tokens, curtok)
        curtok = ''
      end
      -- if the character is the ZERO-WIDTH JOINER character (ZWJ), add joiner
      if v == 0x200D then
        if opt.joiner_annotate and opt.joiner_new and #tokens then
          table.insert(tokens, opt.joiner)
        elseif opt.joiner_annotate then
          if other or (number and unicode.isLetter(nextv)) then
            tokens[#tokens] = tokens[#tokens] .. opt.joiner
          else
            curtok = opt.joiner
          end
        end
      end
      number = false
      letter = false
      space = true
      other = false
    else
      -- skip special characters and BOM and
      if v > 32 and not(v == 0xFEFF) then
        -- normalize the separator marker and feat separator
        if c == separators.joiner_marker then c = separators.joiner_marker_substitute end
        if c == separators.feat_marker then c = separators.feat_marker_substitute end


        local is_letter = unicode.isLetter(v)
        local is_alphabet
        if is_letter and (opt.segment_alphabet_change or #opt.segment_alphabet>0) then
          is_alphabet = alphabet.findAlphabet(v)
        end

        local is_number = unicode.isNumber(v)
        local is_mark = unicode.isMark(v)
        -- if we have a mark, we keep type of previous character
        if is_mark then
          is_letter = letter
          is_number = number
        end
        if opt.mode == 'conservative' then
          if is_number or (c == '-' and letter == true) or c == '_' or
                (letter == true and (c == '.' or c == ',') and (unicode.isNumber(nextv) or unicode.isLetter(nextv))) then
            is_letter = true
          end
        end
        if is_letter then
          if not(letter == true or space == true) or
             (letter == true and not is_mark and
              (prev_alphabet == 'placeholder' or
               (prev_alphabet == is_alphabet and inTable(is_alphabet, opt.segment_alphabet)) or
               (prev_alphabet ~= is_alphabet and opt.segment_alphabet_change))) then
            if opt.joiner_annotate and not(opt.joiner_new) and prev_alphabet ~= 'placeholder' then
              curtok = curtok .. opt.joiner
            end
            table.insert(tokens, curtok)
            if opt.joiner_annotate and opt.joiner_new then
              table.insert(tokens, opt.joiner)
            end
            curtok = ''
            if opt.joiner_annotate and not(opt.joiner_new) and prev_alphabet == 'placeholder' then
              curtok = curtok .. opt.joiner
            end
          elseif other == true then
            if opt.joiner_annotate then
              if curtok == '' then
                if opt.joiner_new then table.insert(tokens, opt.joiner)
                else tokens[#tokens] = tokens[#tokens] .. opt.joiner end
             end
           end
          end
          curtok = curtok .. c
          space = false
          number = false
          other = false
          letter = true
          prev_alphabet = is_alphabet
        elseif is_number then
          if letter == true or not(number == true or space == true) then
            local addjoiner = false
            if opt.joiner_annotate then
              if opt.joiner_new then
                addjoiner = true
              else
                if not(letter) and not(placeholder) then
                  curtok = curtok .. opt.joiner
                else
                  c = opt.joiner .. c
                end
              end
            end
            table.insert(tokens, curtok)
            if addjoiner then
              table.insert(tokens, opt.joiner)
            end
            curtok = ''
          elseif other == true then
            if opt.joiner_annotate then
              if opt.joiner_new then
                table.insert(tokens, opt.joiner)
              else
                tokens[#tokens] = tokens[#tokens] .. opt.joiner
              end
            end
          end
          curtok = curtok..c
          space = false
          letter = false
          other = false
          number = true
        else
          if not space == true then
            if opt.joiner_annotate and not(opt.joiner_new) then
              c = opt.joiner .. c
            end
            table.insert(tokens, curtok)
            if opt.joiner_annotate and opt.joiner_new then
              table.insert(tokens, opt.joiner)
            end
            curtok = ''
          elseif other == true then
            if opt.joiner_annotate then
              if opt.joiner_new then
                table.insert(tokens, opt.joiner)
              else
                curtok = opt.joiner
              end
            end
          end
          curtok = curtok .. c
          table.insert(tokens, curtok)
          curtok = ''
          number = false
          letter = false
          other = true
          space = true
        end
      end
    end
  end

  -- last token
  if (curtok ~= '') then
    table.insert(tokens, curtok)
  end

  return tokens
end
    '''
)
analyseToken = lua.eval(
    '''
function(t, joiner)
  local feats = {}
  local tok = ""
  local p
  local leftsep = false
  local rightsep = false
  local i = 1
  while i <= #t do
    if t:sub(i, i+#separators.feat_marker-1) == separators.feat_marker then
      p = i
      break
    end
    tok = tok .. t:sub(i, i)
    i = i + 1
  end
  if tok:sub(1,#joiner) == joiner then
    tok = tok:sub(1+#joiner)
    leftsep = true
    if tok == '' then rightsep = true end
  end
  if tok:sub(-#joiner,-1) == joiner then
    tok = tok:sub(1,-#joiner-1)
    rightsep = true
  end
  if p then
    p = p + #separators.feat_marker
    local j = p
    while j <= #t do
      if t:sub(j, j+#separators.feat_marker-1) == separators.feat_marker then
        table.insert(feats, t:sub(p, j-1))
        j = j + #separators.feat_marker - 1
        p = j + 1
      end
      j = j + 1
    end
    table.insert(feats, t:sub(p))
  end
  return tok, leftsep, rightsep, feats
end
    '''
)
getTokens = lua.eval(
    '''
function(t, joiner)
  local fields = {}
  t:gsub("([^ ]+)", function(tok)
    local w, leftsep, rightsep, feats =  analyseToken(tok, joiner)
    table.insert(fields, { w=w, leftsep=leftsep, rightsep=rightsep, feats=feats })
  end)
  return fields
end
    '''
)
separators = lua.eval(
    '''
{
  joiner_marker = '￭',
  joiner_marker_substitute = '■',
  feat_marker = '￨',
  feat_marker_substitute = '│',
  BOT = "<w>",
  EOT = "</w>",
  ph_marker_open = '｟',
  ph_marker_close = '｠',
  protected_character = '％'
}
    '''
)
unicode = lua.eval(
    '''
-- for lua < 5.3 compatibility
bit32 = nil
if not bit32 then
  bit32 = require('bit32')
end

unidata = require('tools.utils.unidata')

unicode = {}

-- convert the next utf8 character to ucs
-- returns codepoint and utf-8 character
function unicode._utf8_to_cp(s, idx)
  if idx > #s then return end
  idx = idx or 1
  local c = string.byte(s, idx)
  if c < 0x80 then return c, string.char(c) end
  local l = (c < 0xE0 and 2) or (c < 0xF0 and 3) or (c < 0xF8 and 4)
  if not l then error("invalid utf-8 sequence") end
  local val = bit32.band(c, bit32.rshift(0xff, l))
  for i = 2, l do
    c = string.byte(s, idx+i-1)
    assert(bit32.band(c, 0xC0) == 0x80)
    val = bit32.lshift(val, 6)
    val = bit32.bor(val, bit32.band(c, 0x3F))
  end
  return val, string.sub(s, idx, idx+l-1)
end

-- convert unicode codepoint to utf8
function unicode._cp_to_utf8(u)
  assert(u>=0 and u<=0x10FFFF)
  if u <= 0x7F then
    return string.char(u)
  elseif u <= 0x7FF then
    local b0 = 0xC0 + bit32.rshift(u, 6)
    local b1 = 0x80 + bit32.band(u, 0x3F)
    return string.char(b0, b1)
  elseif u <= 0xFFFF then
    local b0 = 0xE0 + bit32.rshift(u, 12)
    local b1 = 0x80 + bit32.band(bit32.rshift(u, 6), 0x3f)
    local b2 = 0x80 + bit32.band(u, 0x3f)
    return string.char(b0, b1, b2)
  end
  local b0 = 0xF0 + bit32.rshift(u, 18)
  local b1 = 0x80 + bit32.band(bit32.rshift(u, 12), 0x3f)
  local b2 = 0x80 + bit32.band(bit32.rshift(u, 6), 0x3f)
  local b3 = 0x80 + bit32.band(u, 0x3f)
  return string.char(b0, b1, b2, b3)
end

function unicode.utf8_iter(s)
  local L = #s
  local nextv, nextc = unicode._utf8_to_cp(s, 1)
  local p = 1
  if nextc then
    p = p + #nextc
  end
  return function()
    local v, c = nextv, nextc
    if p > L then
      if nextc then
        nextc = nil
        return v, c
      end
      return
    end
    nextv, nextc = unicode._utf8_to_cp(s, p)
    p = p + #nextc
    return v, c, nextv, nextc
  end
end


function unicode.utf8len (s)
  local length = 0
  for _, _ in unicode.utf8_iter(s) do
    length = length + 1
  end
  return length
end

function unicode.utf8substr (s, begin_idx, end_idx)
  local substr = {}
  local idx = 1
  for _, c in unicode.utf8_iter(s) do
    if begin_idx <= idx and idx <= end_idx then
      table.insert(substr, c)
    elseif idx > end_idx then
      break
    end
    idx = idx + 1
  end
  return table.concat(substr, "")
end

function _find_codepoint(u, utable)
  for i,v in pairs(utable) do
    if u >= i then
      local idx = bit32.rshift(u-i,4) + 1
      local p = (u-i) % 16
      if v[idx] then
        return not(bit32.band(bit32.lshift(v[idx], p), 0x8000) == 0)
      end
    end
  end
  return false
end

function unicode.isSeparator(u)
  if not u then return false end
  -- control character or separator
  return u == 32 or (u >= 9 and u <= 13) or _find_codepoint(u, unidata.Separator)
end

function unicode.isMark(u)
  if not u then return false end
  -- control character or separator
  return _find_codepoint(u, unidata.Mark)
end

-- returns if letter and case "lower", "upper", "other"
function unicode.isLetter(u)
  if not u then return false end
  -- accelerate on common ascii
  if u >= 97 and u <= 122 then return true, "lower" end
  if u >= 65 and u <= 90 then return true, "upper" end
  if u <= 127 then return false end
  -- unicode letter or CJK Unified Ideograph
  if ((u>=0x4E00 and u<=0x9FD5) -- CJK Unified Ideograph
      or (u>=0x2F00 and u<=0x2FD5) -- Kangxi Radicals
      or (u>=0x2E80 and u<=0x2EFF) -- CJK Radicals Supplement
      or (u>=0x3040 and u<=0x319F) -- Hiragana, Katakana, Bopomofo, Hangul Compatibility Jamo, Kanbun
      or (u>=0x1100 and u<=0x11FF) -- Hangul Jamo
      or (u>=0xAC00 and u<=0xD7AF) -- Hangul Syllables
      or _find_codepoint(u, unidata.LetterOther)
      ) then
    return true, "other"
  end
  if _find_codepoint(u, unidata.LetterLower) then
    return true, "lower"
  end
  if _find_codepoint(u, unidata.LetterUpper) then
    return true, "upper"
  end
  return false
end

-- convert unicode character to lowercase form if defined in unicodedata
function unicode.getLower(u)
  local l = unidata.maplower[u]
  if l then
    return l, unicode._cp_to_utf8(l)
  end
  return
end

-- convert unicode character to uppercase form if defined in unicodedata
-- dynamically reverse maplower if necessary
function unicode.getUpper(l)
  if not unicode.mapupper then
    -- make sure that reversing, we keep the smallest codepoint because we have İ>i, and I>i
    unidata.mapupper = {}
    for uidx,lidx in pairs(unidata.maplower) do
      if not unidata.mapupper[lidx] or unidata.mapupper[lidx] > uidx then
        unidata.mapupper[lidx] = uidx
      end
    end
  end
  local u = unidata.mapupper[l]
  if u then
    return u, unicode._cp_to_utf8(u)
  end
  return
end

function unicode.isNumber(u)
  if not u then return false end
  -- accelerate on common ascii
  if u >= 48 and u <= 57 then return true end
  if u <= 127 then return false end
  return _find_codepoint(u, unidata.Number)
end

return unicode

    '''
)

import sys, os
__file__ = '/home/jack/Documents/QA_QG/GAN-QA/src/util/'
sys.path.append(os.path.abspath(__file__))
import data_proc
reload(data_proc)
from data_proc import *

dataset = 'squad'
f_name = 'train-v1.1.json'
path_to_dataset = '/home/jack/Documents/QA_QG/data/'
path_to_data = path_to_dataset + dataset + '/' + f_name
raw_triplets = read_raw_squad(path_to_data)

ans = raw_triplets[0][1]
tokenize(ans, )