import pandas as pd
import  matplotlib.pyplot as plt
import cmath
import math
import csv
import numpy as np
from scipy.optimize import curve_fit

data = pd.read_csv('../data/real_im.csv', sep = ';')

def logarithmic_function(x, a, b):
    return a * np.log(x) + b
def sine_function(x, a, b, c, d):
    return a * np.sin(b * x + c) + d

#z = complex(2, 3)
def calculate_reflection_coefficient(s11, s21):
    X  = (s11**2- s21**2 + 1) / (2*s11)
    gamma1 = X+cmath.sqrt(X**2 -1)
    gamma2 = X-cmath.sqrt(X**2 -1)
    if -1*abs(gamma1) >-1 and abs(gamma1) < 1:
        return gamma1
    else:
        return gamma2

def calculate_transmission_coefficient(s11, s21, gamma):
    T = (s11+s21-gamma)/(1-(s11+s21)*gamma)
    return T

def calcualte_premeability(L, T, gamma, lambda_0, lambda_c):
    '''
        L: meters is the length of the sample 
        T: is the transmission coefficient
        gamma: is the reflection coefficient 
    '''
    L_coeff = 1/(2*math.pi*L)
    lambda_sq_inversed = -1*(L_coeff*cmath.log(1/T))**2
    Lambda  = cmath.sqrt(1/lambda_sq_inversed)
    mu_r = (1+gamma)/(Lambda*(1-gamma)*math.sqrt(1/lambda_0**2 - 1/lambda_c**2))
    return mu_r, Lambda

def calculate_permittivity(lambda_0, lambda_c, Lambda, mu_r):
    epsilon_r = (lambda_0**2/mu_r)*(1/ lambda_c**2 + 1/Lambda**2)
    return epsilon_r



lambda_0 = 0.03
lambda_c = 0.0457
L = float(5e-3)
permeability = []
permittivity = []
sum_permittivity = complex(0, 0)
with open('../data/real_im.csv', newline='') as csvfile:
    # Create a CSV reader object
    reader = csv.reader(csvfile, delimiter=';')
    # Loop through each row in the CSV file
    for i, row in enumerate(reader):
        if i==0:
            continue
        s11 = complex(float(row[1]),float(row[2]))
        s21 = complex(float(row[3]),float(row[4]))        
        gamma = calculate_reflection_coefficient(s11, s21)
        T = calculate_transmission_coefficient(s11, s21, gamma)
        #print(T)
        mu_r, Lambda = calcualte_premeability(L, T, gamma, lambda_0, lambda_c)
        permeability.append(mu_r)
        epsilon_r = calculate_permittivity(lambda_0, lambda_0, Lambda, mu_r)
        sum_permittivity+= epsilon_r
        permittivity.append(epsilon_r)

print(len(permeability), len(permittivity))
popt, pcov = curve_fit(sine_function,np.array(data['freq[Hz]']), np.array(permittivity))
plt.plot(np.array(data['freq[Hz]']), np.array(permittivity), label='data')
#plt.plot(np.array(data['freq[Hz]']), sine_function(np.array(data['freq[Hz]']), *popt), label='fit')
plt.legend()
plt.show()
#plt.plot(np.array(data['freq[Hz]']), np.array(permittivity))
#plt.show()

