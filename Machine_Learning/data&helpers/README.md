# Project Overview
In many criminal justice systems around the world, inmates can be released from prison under the parole system prior to completing their sentence if they are deemed not to be a threat to society. They are still considered to be serving their sentence while on parole, and they can be returned to prison if they violate the terms of their parole.
Parole boards are charged with identifying which inmates are good candidates for release on parole. They seek to release inmates who will not commit additional crimes after release. In this problem, we will build and validate a model that predicts if an inmate will violate the terms of his or her parole. Such a model could be useful to a parole board when deciding to approve or deny an application for parole.
For this prediction task, we will use data from the United States 2004 National Corrections Reporting Program; a census across all the United States of parole releases that occurred during 2004. We limited our focus to parolees who served no more than 6 months in prison and whose maximum sentence for all charges did not exceed 18 months. The dataset contains all such parolees who either successfully completed their term of parole during 2004 or those who violated the terms of their parole during that year.


# Problem Statement 
In this project, we will evaluate the performance and predictive power of a model that has been trained and tested on data collected from the United States 2004 National Corrections Reporting Program; a census across all the United States of parole releases that occurred during 2004. We limited our focus to parolees who served no more than 6 months in prison and whose maximum sentence for all charges did not exceed 18 months.
A model trained on this data that is seen as a good fit could then be used to make certain predictions about prisonners who should obtain parole and ultimately solve the problem of prisons overcrowding.

# Data Exploration
 The dataset contains the following variables:
male: 1 if the parolee is male, 0 if female
race: 1 if the parolee is white, 2 otherwise
age: the parolee's age (in years) when he or she was released from prison
state: a code for the parolee's state. 2 is Kentucky, 3 is Louisiana, 4 is Virginia, and 1 is any other state. The three states were selected due to having a high representation in the dataset.
time.served: the number of months the parolee served in prison (limited by the inclusion criteria to not exceed 6 months).
max.sentence: the maximum sentence length for all charges, in months (limited by the inclusion criteria to not exceed 18 months).
multiple.offenses: 1 if the parolee was incarcerated for multiple offenses, 0 otherwise.
crime: a code for the parolee's main crime leading to incarceration. 2 is larceny, 3 is drug-related crime, 4 is driving-related crime, and 1 is any other crime.
violator: 1 if the parolee violated the parole, and 0 if the parolee completed the parole without violation.