
#%%

import io
import numpy as np 
import pandas as pd
from collections import Counter

#%%
full_data_df = pd.read_csv("C:\Mannu\Projects\Sporozoite Spectra for An funestus s.s\Phd data\Analysis data\Biological_attr.dat", delimiter= '\t')

# Select only abdomen data

blood_field_df = full_data_df.query("Cat6 == 'BF' and Cat7 == 'AB'")
print('The shape of field blood meal source data : {}'.format(blood_field_df.shape))

# Observe first few observations
blood_field_df.head()

#%%
# Import PCR results which contains the ID's of blood fed mosquitoes 
pcr_data_df = pd.read_csv("C:\Mannu\Projects\Mannu Phd\MIRS_blood_meal_PCR.csv")
print(pcr_data_df.head(5))

# Now select only human fed samples 
human_df = pcr_data_df.query("PCR_RESULTS == 'Human'")

# Now select only bovine fed samples 
bovine_df = pcr_data_df.query("PCR_RESULTS == 'Bovine'")

# Now select only Dog fed samples 
dog_df = pcr_data_df.query("PCR_RESULTS == 'Dog'")

# Now select only Cow_dog fed samples 
Cow_dog_df = pcr_data_df.query("PCR_RESULTS == 'Cow_dog'")

# Now select only Human_dogfed samples 
Human_dog_df = pcr_data_df.query("PCR_RESULTS == 'Human_dog'")

# Now select only Human_cow fed samples 
Human_cow_df = pcr_data_df.query("PCR_RESULTS == 'Human_cow'")

# Now select only Human_cow fed samples 
unplified_df = pcr_data_df.query("PCR_RESULTS == 'N'")

#%%

# Select a vector of sample ID from PCR data and use it to index all the human blood samples
# from the blood meal field data

human_b_samples_df = blood_field_df[blood_field_df['ID'].isin(list(human_df['SAMPLE_ID']))] # blood_field_df.query("ID in @human_b_samples")

# create a new column in positive samples dataframe and name the samples as positives
human_b_samples_df['blood_meal'] = 'Human'

# Select a vector of sample ID from PCR data and use it to index all the bovine blood samples
# from the blood meal field data

bovine_b_samples_df = blood_field_df[blood_field_df['ID'].isin(list(bovine_df['SAMPLE_ID']))] #query("ID in @bovine_b_samples")

# create a new column in positive samples dataframe and name the samples as positives
bovine_b_samples_df['blood_meal'] = 'Bovine'

# Select a vector of sample ID from PCR data and use it to index all the dog blood samples
# from the blood meal field data

dog_b_samples_df = blood_field_df[blood_field_df['ID'].isin(list(dog_df['SAMPLE_ID']))] # blood_field_df.query("ID in @human_b_samples")

# create a new column in positive samples dataframe and name the samples as positives
dog_b_samples_df['blood_meal'] = 'Dog'

# Select a vector of sample ID from PCR data and use it to index all the Cow_dog blood samples
# from the blood meal field data

Cow_dog_b_samples_df = blood_field_df[blood_field_df['ID'].isin(list(Cow_dog_df['SAMPLE_ID']))] #query("ID in @bovine_b_samples")

# create a new column in positive samples dataframe and name the samples as positives
Cow_dog_b_samples_df['blood_meal'] = 'Cow_dog'

# Select a vector of sample ID from PCR data and use it to index all the Human_dog blood samples
# from the blood meal field data

Human_dog_b_samples_df = blood_field_df[blood_field_df['ID'].isin(list(Human_dog_df['SAMPLE_ID']))] #query("ID in @bovine_b_samples")

# create a new column in positive samples dataframe and name the samples as positives
Human_dog_b_samples_df['blood_meal'] = 'Human_dog'

# Select a vector of sample ID from PCR data and use it to index all the Human_cow blood samples
# from the blood meal field data

Human_cow_b_samples_df = blood_field_df[blood_field_df['ID'].isin(list(Human_cow_df['SAMPLE_ID']))] #query("ID in @bovine_b_samples")

# create a new column in positive samples dataframe and name the samples as positives
Human_cow_b_samples_df['blood_meal'] = 'Human_cow'

unamplified_df = blood_field_df[blood_field_df['ID'].isin(list(unplified_df['SAMPLE_ID']))] #query("ID in @bovine_b_samples")

# create a new column in positive samples dataframe and name the samples as positives
unamplified_df['blood_meal'] = 'N'

# Concatinating human and bovine bloodfed dataframes together

human_bov_bldfed_df = pd.concat([human_b_samples_df, bovine_b_samples_df,
                                dog_b_samples_df, Cow_dog_b_samples_df,
                                Human_dog_b_samples_df, Human_cow_b_samples_df, unamplified_df], axis = 0, join = 'outer')
human_bov_bldfed_df

#%%

# get the metadata from the field data which contain trap type, indoor and outdoor locations and type 
# the mosquito specie

temporary_bf_df = human_bov_bldfed_df
first_column = temporary_bf_df.pop('blood_meal')
  
# insert column using insert(position,column_name,
# first_column) function
temporary_bf_df.insert(1, 'blood_meal', first_column)

blood_meal_metadata_df = temporary_bf_df.iloc[:,1:6]
blood_meal_metadata_df.rename(columns = {'Cat2':'species', 'Cat3':'hh_id', 'blood_meal':'host_blood',
                                         'Cat4':'trap_method', 'Cat5':'position'}, inplace = True) 
blood_meal_metadata_df['trap_method'] = blood_meal_metadata_df['trap_method'].str.replace('RST', 'RBK')
blood_meal_metadata_df['species'] = blood_meal_metadata_df['species'].str.replace('AG', 'arabiensis')
blood_meal_metadata_df['species'] = blood_meal_metadata_df['species'].str.replace('AF', 'funestus')
blood_meal_metadata_df['position'] = blood_meal_metadata_df['position'].str.replace('IN', 'indoor')
blood_meal_metadata_df['position'] = blood_meal_metadata_df['position'].str.replace('OUT', 'outdoor')
blood_meal_metadata_df

# %%
temp_metadata = blood_meal_metadata_df.groupby(['species', 'hh_id', 'position', 'trap_method', 'host_blood']).size()
temp_metadata  = pd.DataFrame(temp_metadata.reset_index())

# save metadata dataframe to disk
temp_metadata.to_csv('C:\Mannu\Projects\Mannu Phd\_blood_meal_metadata.csv')
temp_metadata 

# %%

# load metadata
bf_metadata_df = pd.read_csv('C:\Mannu\Projects\Mannu Phd\_blood_meal_metadata.csv')
bf_metadata_df.rename(columns = {'0':'count'}, inplace = True) 
bf_metadata_df = bf_metadata_df.drop(['Unnamed: 0'], axis = 1)
bf_metadata_df.to_csv('C:\Mannu\Projects\Mannu Phd\_blood_meal_metadata_final.csv')
bf_metadata_df.head(5)

# %%
temp_3 = pd.DataFrame(temp_1.groupby(['Cat2', 'Cat3', 'Cat4', 'Cat5', 'blood_meal']).size().reset_index())

temp_3.rename(columns = {
                            'Cat2':'Species', 
                            'Cat3':'HH ID', 
                            'blood_meal':'Host blood', 
                            'Cat4':'Trap',
                            'Cat5': 'Position'
                        }, 
                    inplace = True) 
                    
temp_3['Trap'] = temp_3['Trap'].str.replace('RST', 'RBK')
temp_3['Species'] = temp_3['Species'].str.replace('AG', 'arabiensis')
temp_3['Species'] = temp_3['Species'].str.replace('AF', 'funestus')
temp_3['Position'] = temp_3['Position'].str.replace('IN', 'Indoor')
temp_3['Position'] = temp_3['Position'].str.replace('OUT', 'Outdoor')

temp_3.to_csv(r'C:\Mannu\Projects\Mannu Phd\Final analysis\Results\Logistic regression\unamplified_test_set_3.csv', index = False)

# %%
temp_4 = pd.DataFrame(temp_2.groupby(['Cat2', 'Cat3', 'Cat4', 'Cat5', 'blood_meal']).size().reset_index())

temp_4.rename(columns = {
                            'Cat2':'Species', 
                            'Cat3':'HH ID', 
                            'blood_meal':'Host blood', 
                            'Cat4':'Trap',
                            'Cat5': 'Position'
                        }, 
                    inplace = True) 
                    
temp_4['Trap'] = temp_4['Trap'].str.replace('RST', 'RBK')
temp_4['Species'] = temp_4['Species'].str.replace('AG', 'arabiensis')
temp_4['Species'] = temp_4['Species'].str.replace('AF', 'funestus')
temp_4['Position'] = temp_4['Position'].str.replace('IN', 'Indoor')
temp_4['Position'] = temp_4['Position'].str.replace('OUT', 'Outdoor')

temp_4.to_csv('C:\Mannu\Projects\Mannu Phd\Final analysis\Results\Logistic regression\imbalanced_test_set_4.csv', index = False)

# %%
