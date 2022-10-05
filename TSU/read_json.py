#!/usr/bin/env python
# coding: utf-8

# In[2]:


import json
import os.path
from sklearn.utils import shuffle

# file = "smarthome_CS_51_old.json"
# first value is user input number of testing they want extract
# second valee is user input number of training they wan extract
# third value is user input what file they want to extract the data from
# fourth value is user input what file they want name their new json file
def get_jsonfile(userinputtesting,userinputtraining,userinputfile,userfilenaming,userpathfilesave):
    file = userinputfile
    # user input both testing and training
    user_selecttesting = userinputtesting
    user_selecttraining = userinputtraining
    user_pathfilesave = userpathfilesave
#    user select file
    f = open(file)
    data = json.load(f)
    testing = []
    training = []
    final_training = []
    final_testing = []
    # Run file to get data name in new array where it is testing or training
    for info in data:
        if data[info]['subset'] == "testing":
            testing.append(info)
        else:
            training.append(info)
    # randomizing the order of array
    testing = shuffle(testing)
    training = shuffle(training)
    # saving the whole data into array for both training and testing
    for i in range(user_selecttesting):
        for datatest in data:
            if datatest == testing[i]:
                result1 = (datatest,data[datatest])
                final_testing.append(result1)
    for i in range(user_selecttraining):
        for datatest in data:
            if datatest == training[i]:
                result = (datatest,data[datatest])
                final_training.append(result)
    # save into a new json file
    with open(userfilenaming+'.json', mode='w+') as f:
        json.dump(final_testing, f, indent=2)
        json.dump(final_training, f, indent=2)
    completeName = os.path.join(user_pathfilesave, userfilenaming+'.json')


# In[7]:


file = "smarthome_CS_51_old.json"
test = "hello"
path = "C:/Users/65912/Documents/GitHub/ict3104-team03-2022/TSU"
get_jsonfile(2,0,file,test,path)


# In[ ]:





# In[ ]:





# In[ ]:




