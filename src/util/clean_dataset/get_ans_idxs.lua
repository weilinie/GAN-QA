--
-- Created by IntelliJ IDEA.
-- User: jack
-- Date: 9/18/17
-- Time: 11:11 AM
-- To change this template use File | Settings | File Templates.
-- functions either copied or adapted from OpenNMT tools folder
--

unicode = require('lua_utils.unicode')
case = require ('lua_utils.case')
separators = require('lua_utils.separators')
alphabet = require('lua_utils.alphabets')

alphabet_list = {}
for k,_ in pairs(alphabet.ranges) do
  table.insert(alphabet_list, k)
end


function inTable(v, t)
  for _, vt in ipairs(t) do
    if v == vt then
      return true
    end
  end
  return false
end

function tokenize(line, opt)
  
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

function tokenizer(opt, line, bpe)
  -- tokenize
  local tokens = tokenize(line, opt)

  -- apply segmetn feature if requested
  if opt.segment_case then
    local sep = ''
    if opt.joiner_annotate then sep = opt.joiner end
    tokens = case.segmentCase(tokens, sep)
  end

  -- apply bpe if requested
  if bpe then
    local sep = ''
    if opt.joiner_annotate then sep = opt.joiner end
    tokens = bpe:segment(tokens, sep)
  end

  -- add-up case feature if requested
  if opt.case_feature then
    tokens = case.addCase(tokens)
  end

  return tokens
end

function analyseToken(t, joiner)
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


function getTokens(t, joiner)
  local fields = {}
  t:gsub("([^ ]+)", function(tok)
    local w, leftsep, rightsep, feats =  analyseToken(tok, joiner)
    table.insert(fields, { w=w, leftsep=leftsep, rightsep=rightsep, feats=feats })
  end)
  return fields
end

function detokenize(line, tokenized_answers_line, ans_start_idx_line, opt)
  dline = ""
  ans_token_start_idx = -1
  ans_token_end_idx = -1
  tokens = getTokens(line, opt.joiner)
  tokens_a = getTokens(tokenized_answers_line, opt.joiner)
  for j = 1, #tokens do
    if ans_start_idx_line == 0 and tokens[j].w == tokens_a[1].w then
      ans_token_start_idx = j
      ans_token_end_idx = table.getn(tokens_a)+ans_token_start_idx-1
      break
    else
      token = tokens[j].w
      if not tokens[j].leftsep and not tokens[j].rightsep then dline = dline .. token .. " "
      elseif not tokens[j].leftsep and tokens[j].rightsep then dline = dline .. token
      elseif tokens[j].leftsep and not tokens[j].rightsep then dline = dline:sub(1,#dline-1) .. token .. " "
      elseif tokens[j].leftsep and tokens[j].rightsep then dline = dline:sub(1,#dline-1) .. token end
  --    dline = dline .. " "
      if utf8.width(dline) == ans_start_idx_line and tokens[j+1].w == tokens_a[1].w then
        ans_token_start_idx = j+1
        ans_token_end_idx = table.getn(tokens_a)+ans_token_start_idx-1
        break
      end
--      print(dline)
--      print(tokens_a[1].w)
--      print(utf8.width(dline))
--      print(ans_start_idx_line)
--      print(j)
--    end
    end
  return dline, ans_token_start_idx, ans_token_end_idx
  end
end
---- test
--i = 60252
--line = t_cs[i]
--tokenized_answers_line = t_as[i]
--ans_start_idx_line = tonumber(new_a_start_idxs_r[i])


-- see if the file exists
function file_exists(file)
  local f = io.open(file, "rb")
  if f then f:close() end
  return f ~= nil
end
-- get all lines from a file, returns an empty
-- list/table if the file does not exist
function lines_from(file)
  if not file_exists(file) then return {} end
  lines = {}
  for line in io.lines(file) do
    lines[#lines + 1] = line
  end
  return lines
end

-- script to read the new json file only english no escape chars and compare if the answers are equal
new_triplets = json.load('/home/jack/Documents/QA_QG/data/processed_squad/dev_squad_EnglishOnly_noEscape.json')
for k = 1, #new_triplets['contexts'] do
  if new_triplets['contexts'][k]:sub(new_triplets['ans_start_idx'][k]+1, new_triplets['ans_end_idx'][k]) ~= new_triplets['answers'][k] then
    print(k)
  end
end
new_cs = new_triplets['contexts']
new_qs = new_triplets['questions']
new_as = new_triplets['answers']
new_a_start_idxs = new_triplets['ans_start_idx']
new_a_end_idxs = new_triplets['ans_end_idx']
-- write line by line
-- have to do this in separate loops otherwise write to file will mess up
new_cs_f ,err = io.open("/home/jack/Documents/QA_QG/data/processed_squad/dev_contexts_EnglishOnly_noEscape.txt","w")
for i = 1, #new_cs do new_cs_f:write(new_cs[i]..'\n') end new_cs_f:close()
new_qs_f ,err = io.open("/home/jack/Documents/QA_QG/data/processed_squad/dev_questions_EnglishOnly_noEscape.txt","w")
for i = 1, #new_qs do new_qs_f:write(new_qs[i]..'\n') end new_qs_f:close()
new_as_f ,err = io.open("/home/jack/Documents/QA_QG/data/processed_squad/dev_answers_EnglishOnly_noEscape.txt","w")
for i = 1, #new_as do new_as_f:write(new_as[i]..'\n') end new_as_f:close()
new_a_start_idxs_f ,err = io.open("/home/jack/Documents/QA_QG/data/processed_squad/dev_a_start_idxs_EnglishOnly_noEscape.txt","w")
for i = 1, #new_a_start_idxs do new_a_start_idxs_f:write(new_a_start_idxs[i]..'\n') end new_a_start_idxs_f:close()
new_a_end_idxs_f ,err = io.open("/home/jack/Documents/QA_QG/data/processed_squad/dev_a_end_idxs_EnglishOnly_noEscape.txt","w")
for i = 1, #new_a_end_idxs do new_a_end_idxs_f:write(new_a_end_idxs[i]..'\n')  end new_a_end_idxs_f:close()
-- sanity check: read new triplets that dont contain escape characters and are only english
new_cs_r = lines_from('/home/jack/Documents/QA_QG/data/processed_squad/dev_contexts_EnglishOnly_noEscape.txt')
new_qs_r = lines_from('/home/jack/Documents/QA_QG/data/processed_squad/dev_questions_EnglishOnly_noEscape.txt')
new_as_r = lines_from('/home/jack/Documents/QA_QG/data/processed_squad/dev_answers_EnglishOnly_noEscape.txt')
new_a_start_idxs_r = lines_from('/home/jack/Documents/QA_QG/data/processed_squad/dev_a_start_idxs_EnglishOnly_noEscape.txt')
new_a_end_idxs_r = lines_from('/home/jack/Documents/QA_QG/data/processed_squad/dev_a_end_idxs_EnglishOnly_noEscape.txt')
-- sanity check to ensure the data read and the data before write are the same
for i = 1, #new_cs_r do
  if new_triplets['ans_end_idx'][i] ~= tonumber(new_a_end_idxs[i]) then print(i) end
end
-- sanity check if the answer char indices matches with the real answers in read files
for i = 1, #new_cs_r do
  if new_cs_r[i]:sub(tonumber(new_a_start_idxs[i])+1, tonumber(new_a_end_idxs[i])) ~= new_as_r[i] then print(i) end
end

-- tokenize:
-- 1. cd into openNMT folder
-- 2. run th tools/tokenize.lua -joiner_annotate false <path_to_file_to_be_tokenized> output_file_path
-- 3. can change the joiner_annotate flag to switch between annotate or no annotate

-- script to read the tokenized 
cs = lines_from('/home/jack/Documents/QA_QG/data/squad_openNMT/contexts_EnglishOnly_noEscape_NoAnnotate.txt')
qs = lines_from('/home/jack/Documents/QA_QG/data/squad_openNMT/questions_EnglishOnly_noEscape_NoAnnotate.txt')
as = lines_from('/home/jack/Documents/QA_QG/data/squad_openNMT/answers_EnglishOnly_noEscape_NoAnnotate.txt')
t_cs = lines_from('/home/jack/Documents/QA_QG/data/squad_openNMT/preparation/dev/tokenized_annotate/dev_contexts_EnglishOnly_noEscape_annotate.txt')
t_qs = lines_from('/home/jack/Documents/QA_QG/data/squad_openNMT/preparation/dev/tokenized_annotate/dev_questions_EnglishOnly_noEscape_annotate.txt')
t_as = lines_from('/home/jack/Documents/QA_QG/data/squad_openNMT/preparation/dev/tokenized_annotate/dev_answers_EnglishOnly_noEscape_annotate.txt')
-- run the detokenizer and record answer token start and end indices
opt = {}
opt.joiner = separators.joiner_marker
opt.case_feature = false
ans_token_start_idxs = {} ans_token_end_idxs = {}
for i = 1, #t_cs do
  print(i)
  dline, ans_token_start_idx, ans_token_end_idx = detokenize(t_cs[i], t_as[i], tonumber(new_a_start_idxs_r[i]), opt)
  ans_token_start_idxs[i] = ans_token_start_idx
  ans_token_end_idxs[i] = ans_token_end_idx
end
-- save file
ans_token_start_idxs_f,err = io.open("/home/jack/Documents/QA_QG/data/squad_openNMT/preparation/dev/dev_ans_token_start_idxs.txt","w")
for i = 1, #ans_token_start_idxs do ans_token_start_idxs_f:write(ans_token_start_idxs[i]..'\n') end ans_token_start_idxs_f:close()
ans_token_end_idxs_f,err = io.open("/home/jack/Documents/QA_QG/data/squad_openNMT/preparation/dev/dev_ans_token_end_idxs.txt","w")
for i = 1, #ans_token_end_idxs do ans_token_end_idxs_f:write(ans_token_end_idxs[i]..'\n') end ans_token_end_idxs_f:close()
-- sanity check: read those files
atsis_temp = lines_from("/home/jack/Documents/QA_QG/data/squad_openNMT/dev/dev_ans_token_start_idxs.txt")
--atsis = {} for i in 1, #atsis_temp do atsis[i] = tonumber(atsis_temp[i]) end
ateis_temp = lines_from("/home/jack/Documents/QA_QG/data/squad_openNMT/dev/dev_ans_token_end_idxs.txt")
--ateis = {} for i in 1, #ateis_temp do ateis[i] = tonumber(ateis_temp[i]) end
-- sanity check: check whether any start token is not found
mismatch_idx = {}
mismatch = {}
for i = 1, #t_cs do
  if ans_token_start_idxs[i] == -1 or ans_token_start_idxs[i] == nil then
    mismatch_idx[#mismatch_idx+1] = i
    mismatch[#mismatch+1] = ans_token_start_idxs[i]
  end
end
-- save those that are NOT mismatch to file
cs_min_annotate = io.open("/home/jack/Documents/QA_QG/data/squad_openNMT/dev/dev_cs_min_annotate.txt", 'w')
for i = 1, #t_cs do
  if ans_token_start_idxs[i] ~= -1 and ans_token_start_idxs[i] ~= nil then
    cs_min_annotate:write(t_cs[i]..'\n')
  end
end
cs_min_annotate:close()
cs_min_NoAnnotate = io.open("/home/jack/Documents/QA_QG/data/squad_openNMT/dev/dev_cs_min_NoAnnotate.txt", 'w')
for i = 1, #cs do
  if ans_token_start_idxs[i] ~= -1 and ans_token_start_idxs[i] ~= nil then
    cs_min_NoAnnotate:write(cs[i]..'\n')
  end
end
cs_min_NoAnnotate:close()
qs_min_annotate = io.open("/home/jack/Documents/QA_QG/data/squad_openNMT/dev/dev_qs_min_annotate.txt", 'w')
for i = 1, #t_qs do
  if ans_token_start_idxs[i] ~= -1 and ans_token_start_idxs[i] ~= nil then
    qs_min_annotate:write(t_qs[i]..'\n')
  end
end
qs_min_annotate:close()
qs_min_NoAnnotate = io.open("/home/jack/Documents/QA_QG/data/squad_openNMT/dev/dev_qs_min_NoAnnotate.txt", 'w')
for i = 1, #qs do
  if ans_token_start_idxs[i] ~= -1 and ans_token_start_idxs[i] ~= nil then
    qs_min_NoAnnotate:write(qs[i]..'\n')
  end
end
qs_min_NoAnnotate:close()
as_min_annotate = io.open("/home/jack/Documents/QA_QG/data/squad_openNMT/dev/dev_as_min_annotate.txt", 'w')
for i = 1, #t_as do
  if ans_token_start_idxs[i] ~= -1 and ans_token_start_idxs[i] ~= nil then
    as_min_annotate:write(t_as[i]..'\n')
  end
end
as_min_annotate:close()
as_min_NoAnnotate = io.open("/home/jack/Documents/QA_QG/data/squad_openNMT/dev/dev_as_min_NoAnnotate.txt", 'w')
for i = 1, #as do
  if ans_token_start_idxs[i] ~= -1 and ans_token_start_idxs[i] ~= nil then
    as_min_NoAnnotate:write(as[i]..'\n')
  end
end
as_min_NoAnnotate:close()
atsi_min = io.open("/home/jack/Documents/QA_QG/data/squad_openNMT/dev/dev_atsi_min.txt", 'w')
for i = 1, #atsis_temp do
  if ans_token_start_idxs[i] ~= -1 and ans_token_start_idxs[i] ~= nil then
    atsi_min:write(ans_token_start_idxs[i]..'\n')
  end
end
atsi_min:close()
atei_min = io.open("/home/jack/Documents/QA_QG/data/squad_openNMT/dev/dev_atei_min.txt", 'w')
for i = 1, #ateis_temp do
  if ans_token_start_idxs[i] ~= -1 and ans_token_start_idxs[i] ~= nil then
    atei_min:write(ans_token_end_idxs[i]..'\n')
  end
end
atei_min:close()
mismatch_idx = {}
for i = 1, #t_cs do
  tokens = getTokens(t_cs[i], opt.joiner)
  tokens_a = getTokens(t_as[i], opt.joiner)
  if tokens[ans_token_start_idxs[i]].w ~= tokens_a[1] then mismatch_idx[#mismatch_idx+1] = i end
end