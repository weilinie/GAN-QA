######################################################################
# Plotting results
# ----------------
#
# Plotting is done with matplotlib, using the array of loss values
# ``plot_losses`` saved while training.
#
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import difflib


def showPlot(points):
    plt.figure()
    fig, ax = plt.subplots()
    # this locator puts ticks at regular intervals
    loc = ticker.MultipleLocator(base=0.2)
    ax.yaxis.set_major_locator(loc)
    plt.plot(points)



def extract(v):
    return v.data.storage().tolist()



######################################################################
# This is a helper function to print time elapsed and estimated time
# remaining given the current time and progress %.
#

import time
import math

def asMinutes(s):
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

def timeSince(since, percent):
    now = time.time()
    s = now - since
    es = s / (percent)
    rs = es - s
    return '%s (- %s)' % (asMinutes(s), asMinutes(rs))



######################################################################
# show loss function
def plotLoss(loss_f, plot_every, save_path=None, from_file=True, f_name='loss.png', title='training loss'):
    if from_file:
        loss_vec = []
        with open(loss_f) as f:
            content = f.readlines()
            content = [x.strip() for x in content] # list of every line, each a string
            for line in content:
                try:
                    loss_vec.append(float(line))
                except ValueError:
                    pass
    else:
        loss_vec = loss_f
    # plot
    plt.figure()
    plt.title(title)
    plt.xlabel('training iterations')
    plt.ylabel('loss')
    plt.grid()
    plt.plot([x*plot_every for x in range(1, len(loss_vec)+1)], loss_vec)
    if save_path == None:
        plt.savefig(f_name)
    else:
        plt.savefig(save_path + '/' + f_name)

# test
# from util import *
# plotLoss('../../../exp_results_temp/G_c_a_sep_pretrain_exp_0902/loss_temp.txt', 30)


######################################################################
# check if the generated question already exist in the corpus
def generated_q_novelty(triplets, generated_q):
    # input - tokenized triplets, each one a list of strings
    # input - generated question
    # output - a similarity score vector for each of the questions in the triplets
    questions = triplets[1]
    scores = []
    if not (isinstance(generated_q, str) or isinstance(generated_q, unicode)):
        generated_q = generated_q.join(' ')
    for q in questions:
        print(q)
        q = q.join(' ')
        scores.append(difflib.SequenceMatcher(None, generated_q, q).ratio)
    return np.array(scores)
# test


# ######################################################################
# # For a better viewing experience we will do the extra work of adding axes
# # and labels:
# #
# def showAttention(input_sentence, output_words, attentions):
#     # Set up figure with colorbar
#     fig = plt.figure()
#     ax = fig.add_subplot(111)
#     cax = ax.matshow(attentions.numpy(), cmap='bone')
#     fig.colorbar(cax)

#     # Set up axes
#     ax.set_xticklabels([''] + input_sentence.split(' ') +
#                        ['<EOS>'], rotation=90)
#     ax.set_yticklabels([''] + output_words)

#     # Show label at every tick
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
#     ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

#     plt.show()


# def evaluateAndShowAttention(input_sentence):
#     output_words, attentions = evaluate(
#         encoder1, attn_decoder1, input_sentence)
#     print('input =', input_sentence)
#     print('output =', ' '.join(output_words))
#     showAttention(input_sentence, output_words, attentions)


