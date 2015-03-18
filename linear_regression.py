import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import statsmodels.api as sm

#loansData = pd.read_csv('https://spark-public.s3.amazonaws.com/dataanalysis/loansData.csv')
loansData = pd.read_csv('loan_data.csv')
print loansData['Interest.Rate'][0:5]

clean_interest_rate = loansData['Interest.Rate'].map(lambda x: round(float(x.rstrip('%'))/100, 4))
print clean_interest_rate[0:5]
loansData['Interest.Rate'] = clean_interest_rate

print loansData['Loan.Length'][0:5]
clean_loan_length = loansData['Loan.Length'].map(lambda x: int(x.rstrip(' months')))
print clean_loan_length[0:5]
loansData['Loan.Length'] = clean_loan_length

print loansData['FICO.Range'][0:5]
score = loansData['FICO.Range'].map(lambda x: x.split('-'))
print score[0:5]
score = score.map(lambda x: int(x[0]))
print score[0:5]

loansData['FICO.Score'] = score
print loansData['FICO.Score'][0:5]

plt.figure()
p = loansData['FICO.Score'].hist()
plt.show()

#plt.figure()
a = pd.scatter_matrix(loansData, alpha=0.05, figsize=(10,10))
plt.show()

#plt.figure()
a = pd.scatter_matrix(loansData, alpha=0.05, figsize=(10,10), diagonal='hist')
plt.show()

intrate = loansData['Interest.Rate']
loanamt = loansData['Amount.Requested']
fico = loansData['FICO.Score']


# The dependent variable
y = np.matrix(intrate).transpose()
# The independent variables shaped as columns
x1 = np.matrix(fico).transpose()
x2 = np.matrix(loanamt).transpose()

x = np.column_stack([x1,x2])
print x[0:5]

X = sm.add_constant(x)
print X
model = sm.OLS(y,X)
f = model.fit()

print 'Coefficients: ', f.params[0:2]
print 'Intercept: ', f.params[2]
print 'P-Values: ', f.pvalues
print 'R-Squared: ', f.rsquared

#Used for logistic regression lesson
loansData.to_csv('loansData_clean.csv', header=True, index=False)
