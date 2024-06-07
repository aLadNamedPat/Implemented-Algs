import numpy as np

sample_times = 100

probabilities = [0.5, 0.3, 0.9, 0.4]
#Bernoulli Thompson Sampling

s_f = [[0, 0],[0, 0],[0, 0], [0, 0]]

for i in range(sample_times):
    #Sample the model here:
    s = [0, 0, 0, 0]
    r = 0
    for i in range(len(probabilities)):
        #Sample the probability of obtaining a reward
        s[i] = np.random.beta(s_f[i][0] + 1, s_f[i][1] + 1)    
    
    #Obtain the argmax of the beta function to determine which lever has the highest probability of return positive reward
    x_t = np.argmax(s)
    if np.random.random() < probabilities[x_t]:
        r = 1
    else:
        r = 0
    s_f[x_t] = [s_f[x_t][0] + r, s_f[x_t][1] + 1 - r]


print(s_f)

