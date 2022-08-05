

# REARRANGE: An Effort Estimation Approach for Software Clustering-based Remodularisation

## Abstract

Software clustering is often used as a remodularisation technique to suggest ways to improve the internal quality of the software through some suggested refactoring operations. This project aims to provide an end-to-end pipeline to help developers in carrying out refactoring activities through the following steps:


1. Estimate the effort needed to convert the current project structure to the suggested clustering result. (**REARRANGE**)

## Introduction

This repo contains scripts and notebooks that are used in REARRANGE. This README shows the general flow of the end-to-end pipeline. For more details and the full implementation, kindly refer back to the specific notebooks.

## REARRANGE Experiment Design
1. Software Systems
2. Identifying Refactoring Operations
3. Proxy Measure for Refactoring Operations
4. Data Preparation
5. Building and Training Models
6. Model Validation
7. Estimation Techniques


```python
import json
import pandas as pd
import numpy as np
import networkx as nx
import jellyfish
import os
import shutil
import subprocess
import requests
from github import Github
from git import Repo
from scipy.cluster.hierarchy import dendrogram, linkage
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score
from sklearn import preprocessing
from sklearn.cluster import AgglomerativeClustering
from zipfile import ZipFile
from filecmp import dircmp
import configparser
```

## 1. Software Systems 

**REARRANGE 01 - Crawl Github Commits.ipynb**

The main function of this notebook is to crawl github commits for each release to obtain the following data from the selected dataset of software systems.

**Inputs**: 
1. Project Name

**Outputs**: 
1. Project Release Name
2. Project Release Commit SHA

A sample of the data is given below.


```python
github_commits_df = pd.read_csv('volatile_projects_complete_links_limit10_filtered.csv')
github_commits_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>project_name</th>
      <th>project_link</th>
      <th>version_name</th>
      <th>commit</th>
      <th>timestamp</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Dbeaver</td>
      <td>https://github.com/dbeaver/dbeaver</td>
      <td>21.1.4</td>
      <td>Commit(sha="113a0a672f277a6e8181757a0c54f92d42...</td>
      <td>29/7/2021 11:08</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Dbeaver</td>
      <td>https://github.com/dbeaver/dbeaver</td>
      <td>21.1.3</td>
      <td>Commit(sha="4430459a3fe06c6140aa40b71ddc41ddf8...</td>
      <td>15/7/2021 8:06</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Dbeaver</td>
      <td>https://github.com/dbeaver/dbeaver</td>
      <td>21.1.2</td>
      <td>Commit(sha="b0693d44048a9c50e750b6df69cfe83fcb...</td>
      <td>2/7/2021 13:34</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Dbeaver</td>
      <td>https://github.com/dbeaver/dbeaver</td>
      <td>21.1.1</td>
      <td>Commit(sha="073dfc26c7a065f5d5abf18be8cce8258a...</td>
      <td>18/6/2021 13:50</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Dbeaver</td>
      <td>https://github.com/dbeaver/dbeaver</td>
      <td>21.1.0</td>
      <td>Commit(sha="17ce2d14317b1160ec9480da549028d182...</td>
      <td>28/5/2021 5:16</td>
    </tr>
  </tbody>
</table>
</div>



## 2. Identifying Refactoring Operations 

**REARRANGE 02 - Crawl Refactoring Miner Data.ipynb**

The main function of this notebooks is to run Refactoring Miner for each project, version and commit. This is to obtain any and all refactoring details between the current commit and the previous commit.


**Inputs**:
1. Project Name
2. Release Name
3. Commit SHA (Current Release)
4. Commit SHA (Previous Release)

**Outputs**:
1. Number of Refactoring
2. Type of Refactoring
3. Location of Refactoring (Previous Location)
    * Start Line Number
    * End Line Number
4. Location of Refactoring (New Location)
    * Start Line Number
    * End Line Number
    
A sample of the data is given below.


```python
refactoring_miner_filename = f'raw_refactoringMiner/Okhttp/Okhttp_parent-5.0.0-alpha.2.json'
print(refactoring_miner_filename)
f = open(refactoring_miner_filename)
refactoring_miner = json.load(f)
for i in refactoring_miner['commits']:
    if len(i['refactorings']) > 0:
        print(i)
        break
```

    raw_refactoringMiner/Okhttp/Okhttp_parent-5.0.0-alpha.2.json
    {'repository': 'https://github.com/square/okhttp', 'sha1': '3e331c108905a97fa9718b40844ddc1356fc86b5', 'url': 'https://github.com/square/okhttp/commit/3e331c108905a97fa9718b40844ddc1356fc86b5', 'refactorings': [{'type': 'Move Class', 'description': 'Move Class okhttp3.mockwebserver.CustomDispatcherTest moved to mockwebserver3.CustomDispatcherTest', 'leftSideLocations': [{'filePath': 'mockwebserver/src/test/java/okhttp3/mockwebserver/CustomDispatcherTest.java', 'startLine': 31, 'endLine': 98, 'startColumn': 0, 'endColumn': 2, 'codeElementType': 'TYPE_DECLARATION', 'description': 'original type declaration', 'codeElement': 'okhttp3.mockwebserver.CustomDispatcherTest'}], 'rightSideLocations': [{'filePath': 'mockwebserver/src/test/java/mockwebserver3/CustomDispatcherTest.java', 'startLine': 31, 'endLine': 98, 'startColumn': 0, 'endColumn': 2, 'codeElementType': 'TYPE_DECLARATION', 'description': 'moved type declaration', 'codeElement': 'mockwebserver3.CustomDispatcherTest'}]}, {'type': 'Move Class', 'description': 'Move Class okhttp3.mockwebserver.RecordedRequestTest moved to mockwebserver3.RecordedRequestTest', 'leftSideLocations': [{'filePath': 'mockwebserver/src/test/java/okhttp3/mockwebserver/RecordedRequestTest.java', 'startLine': 32, 'endLine': 122, 'startColumn': 0, 'endColumn': 2, 'codeElementType': 'TYPE_DECLARATION', 'description': 'original type declaration', 'codeElement': 'okhttp3.mockwebserver.RecordedRequestTest'}], 'rightSideLocations': [{'filePath': 'mockwebserver/src/test/java/mockwebserver3/RecordedRequestTest.java', 'startLine': 32, 'endLine': 122, 'startColumn': 0, 'endColumn': 2, 'codeElementType': 'TYPE_DECLARATION', 'description': 'moved type declaration', 'codeElement': 'mockwebserver3.RecordedRequestTest'}]}, {'type': 'Move Class', 'description': 'Move Class okhttp3.mockwebserver.internal.http2.Http2Server moved to mockwebserver3.internal.http2.Http2Server', 'leftSideLocations': [{'filePath': 'mockwebserver/src/test/java/okhttp3/mockwebserver/internal/http2/Http2Server.java', 'startLine': 46, 'endLine': 193, 'startColumn': 0, 'endColumn': 2, 'codeElementType': 'TYPE_DECLARATION', 'description': 'original type declaration', 'codeElement': 'okhttp3.mockwebserver.internal.http2.Http2Server'}], 'rightSideLocations': [{'filePath': 'mockwebserver/src/test/java/mockwebserver3/internal/http2/Http2Server.java', 'startLine': 46, 'endLine': 193, 'startColumn': 0, 'endColumn': 2, 'codeElementType': 'TYPE_DECLARATION', 'description': 'moved type declaration', 'codeElement': 'mockwebserver3.internal.http2.Http2Server'}]}, {'type': 'Move Class', 'description': 'Move Class okhttp3.mockwebserverwrapper.MockWebServerTest moved to mockwebserver3.MockWebServerTest', 'leftSideLocations': [{'filePath': 'mockwebserverwrapper/src/test/java/okhttp3/mockwebserverwrapper/MockWebServerTest.java', 'startLine': 63, 'endLine': 629, 'startColumn': 0, 'endColumn': 2, 'codeElementType': 'TYPE_DECLARATION', 'description': 'original type declaration', 'codeElement': 'okhttp3.mockwebserverwrapper.MockWebServerTest'}], 'rightSideLocations': [{'filePath': 'mockwebserver/src/test/java/mockwebserver3/MockWebServerTest.java', 'startLine': 63, 'endLine': 629, 'startColumn': 0, 'endColumn': 2, 'codeElementType': 'TYPE_DECLARATION', 'description': 'moved type declaration', 'codeElement': 'mockwebserver3.MockWebServerTest'}]}, {'type': 'Merge Package', 'description': 'Merge Package [okhttp3.mockwebserver, okhttp3.mockwebserverwrapper] to mockwebserver3', 'leftSideLocations': [{'filePath': 'mockwebserver/src/test/java/okhttp3/mockwebserver/CustomDispatcherTest.java', 'startLine': 31, 'endLine': 98, 'startColumn': 0, 'endColumn': 2, 'codeElementType': 'TYPE_DECLARATION', 'description': 'original type declaration', 'codeElement': 'okhttp3.mockwebserver.CustomDispatcherTest'}, {'filePath': 'mockwebserver/src/test/java/okhttp3/mockwebserver/RecordedRequestTest.java', 'startLine': 32, 'endLine': 122, 'startColumn': 0, 'endColumn': 2, 'codeElementType': 'TYPE_DECLARATION', 'description': 'original type declaration', 'codeElement': 'okhttp3.mockwebserver.RecordedRequestTest'}, {'filePath': 'mockwebserver/src/test/java/okhttp3/mockwebserver/internal/http2/Http2Server.java', 'startLine': 46, 'endLine': 193, 'startColumn': 0, 'endColumn': 2, 'codeElementType': 'TYPE_DECLARATION', 'description': 'original type declaration', 'codeElement': 'okhttp3.mockwebserver.internal.http2.Http2Server'}, {'filePath': 'mockwebserverwrapper/src/test/java/okhttp3/mockwebserverwrapper/MockWebServerTest.java', 'startLine': 63, 'endLine': 629, 'startColumn': 0, 'endColumn': 2, 'codeElementType': 'TYPE_DECLARATION', 'description': 'original type declaration', 'codeElement': 'okhttp3.mockwebserverwrapper.MockWebServerTest'}], 'rightSideLocations': [{'filePath': 'mockwebserver/src/test/java/mockwebserver3/CustomDispatcherTest.java', 'startLine': 31, 'endLine': 98, 'startColumn': 0, 'endColumn': 2, 'codeElementType': 'TYPE_DECLARATION', 'description': 'moved type declaration', 'codeElement': 'mockwebserver3.CustomDispatcherTest'}, {'filePath': 'mockwebserver/src/test/java/mockwebserver3/RecordedRequestTest.java', 'startLine': 32, 'endLine': 122, 'startColumn': 0, 'endColumn': 2, 'codeElementType': 'TYPE_DECLARATION', 'description': 'moved type declaration', 'codeElement': 'mockwebserver3.RecordedRequestTest'}, {'filePath': 'mockwebserver/src/test/java/mockwebserver3/internal/http2/Http2Server.java', 'startLine': 46, 'endLine': 193, 'startColumn': 0, 'endColumn': 2, 'codeElementType': 'TYPE_DECLARATION', 'description': 'moved type declaration', 'codeElement': 'mockwebserver3.internal.http2.Http2Server'}, {'filePath': 'mockwebserver/src/test/java/mockwebserver3/MockWebServerTest.java', 'startLine': 63, 'endLine': 629, 'startColumn': 0, 'endColumn': 2, 'codeElementType': 'TYPE_DECLARATION', 'description': 'moved type declaration', 'codeElement': 'mockwebserver3.MockWebServerTest'}]}, {'type': 'Move Source Folder', 'description': 'Move Source Folder mockwebserver to mockwebserverwrapper', 'leftSideLocations': [], 'rightSideLocations': []}]}
    

## 3. Proxy Measure for Refactoring Operations & 4. Data Preparation

**REARRANGE 03 - Merge Data (Github Commit, Refactoring Miner, Depends, CKMetrics).ipynb**

The main function of this notebooks is to 
1. Merge the data from the previous 2 notebooks with dependency features (Depends) and software features (CKMetrics).
2. Calculate the proxy measure for refactoring operations given by refactoring loc / total loc in commit.
3. Calculate the effort needed for other Sofware Estimation Models
    * COCOMOII
    * GeneticP
    * SoftwareMaintenance
    * Fuzzy


**Inputs**:
1. Github Commit Data
2. Refactoring Miner Data
3. Depends Data
4. CKMetrics Data

**Outputs**:
1. Main DataFrame
    
A sample of the data is given below.


```python
effort_estimation_df = pd.read_csv('Effort_Estimation_Results_3E_v2/Okhttp.csv')
effort_estimation_df.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>kmean_label</th>
      <th>time_taken_mean</th>
      <th>time_taken_min</th>
      <th>time_taken_max</th>
      <th>time_taken_q10</th>
      <th>time_taken_q20</th>
      <th>time_taken_q25</th>
      <th>time_taken_q30</th>
      <th>time_taken_q40</th>
      <th>time_taken_q50</th>
      <th>...</th>
      <th>actual_num_of_classes_touched_min</th>
      <th>actual_num_of_classes_touched_max</th>
      <th>actual_num_of_classes_touched_std</th>
      <th>commit_line_changed</th>
      <th>refactoring_perc</th>
      <th>refactoring_perc_time_taken</th>
      <th>cocomoII_time_taken</th>
      <th>geneticP_time_taken</th>
      <th>softwareMaintenance_time_taken</th>
      <th>fuzzy_time_taken</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>5</td>
      <td>25.302839</td>
      <td>1.0</td>
      <td>167.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>8.0</td>
      <td>11.0</td>
      <td>...</td>
      <td>2</td>
      <td>40</td>
      <td>15.649814</td>
      <td>1182</td>
      <td>0.060068</td>
      <td>1.000000</td>
      <td>611.61408</td>
      <td>364.998402</td>
      <td>7713.92</td>
      <td>1176.156883</td>
    </tr>
    <tr>
      <th>1</th>
      <td>5</td>
      <td>25.302839</td>
      <td>1.0</td>
      <td>167.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>8.0</td>
      <td>11.0</td>
      <td>...</td>
      <td>39</td>
      <td>82</td>
      <td>21.733231</td>
      <td>1000</td>
      <td>0.344000</td>
      <td>3.096000</td>
      <td>517.44000</td>
      <td>308.943360</td>
      <td>6520.00</td>
      <td>1052.384959</td>
    </tr>
    <tr>
      <th>2</th>
      <td>5</td>
      <td>25.302839</td>
      <td>1.0</td>
      <td>167.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>8.0</td>
      <td>11.0</td>
      <td>...</td>
      <td>2</td>
      <td>2</td>
      <td>NaN</td>
      <td>338</td>
      <td>1.000000</td>
      <td>146.000000</td>
      <td>174.89472</td>
      <td>104.602433</td>
      <td>2177.28</td>
      <td>511.569022</td>
    </tr>
    <tr>
      <th>3</th>
      <td>5</td>
      <td>25.302839</td>
      <td>1.0</td>
      <td>167.0</td>
      <td>1.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>5.0</td>
      <td>8.0</td>
      <td>11.0</td>
      <td>...</td>
      <td>1</td>
      <td>8</td>
      <td>2.366432</td>
      <td>14</td>
      <td>1.000000</td>
      <td>1.000000</td>
      <td>7.24416</td>
      <td>4.336286</td>
      <td>51.84</td>
      <td>61.567253</td>
    </tr>
    <tr>
      <th>4</th>
      <td>7</td>
      <td>24.024272</td>
      <td>1.0</td>
      <td>148.0</td>
      <td>2.0</td>
      <td>4.0</td>
      <td>5.0</td>
      <td>6.5</td>
      <td>10.0</td>
      <td>14.0</td>
      <td>...</td>
      <td>2</td>
      <td>48</td>
      <td>19.605194</td>
      <td>358</td>
      <td>0.360335</td>
      <td>1.081006</td>
      <td>185.24352</td>
      <td>110.786180</td>
      <td>2308.48</td>
      <td>531.504377</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 206 columns</p>
</div>




```python
for column in effort_estimation_df:
    print(column)
```

    kmean_label
    time_taken_mean
    time_taken_min
    time_taken_max
    time_taken_q10
    time_taken_q20
    time_taken_q25
    time_taken_q30
    time_taken_q40
    time_taken_q50
    time_taken_q60
    time_taken_q70
    time_taken_q75
    time_taken_q80
    time_taken_q90
    sha
    name
    email
    date
    login
    message
    parent_sha
    parent_date
    time_taken
    contains_refactoring
    project_name
    commit_compared_with
    cbo_mean
    cbo_min
    cbo_max
    cbo_std
    wmc_mean
    wmc_min
    wmc_max
    wmc_std
    dit_mean
    dit_min
    dit_max
    dit_std
    rfc_mean
    rfc_min
    rfc_max
    rfc_std
    lcom_mean
    lcom_min
    lcom_max
    lcom_std
    totalMethods_mean
    totalMethods_min
    totalMethods_max
    totalMethods_std
    staticMethods_mean
    staticMethods_min
    staticMethods_max
    staticMethods_std
    publicMethods_mean
    publicMethods_min
    publicMethods_max
    publicMethods_std
    privateMethods_mean
    privateMethods_min
    privateMethods_max
    privateMethods_std
    protectedMethods_mean
    protectedMethods_min
    protectedMethods_max
    protectedMethods_std
    defaultMethods_mean
    defaultMethods_min
    defaultMethods_max
    defaultMethods_std
    abstractMethods_mean
    abstractMethods_min
    abstractMethods_max
    abstractMethods_std
    finalMethods_mean
    finalMethods_min
    finalMethods_max
    finalMethods_std
    synchronizedMethods_mean
    synchronizedMethods_min
    synchronizedMethods_max
    synchronizedMethods_std
    totalFields_mean
    totalFields_min
    totalFields_max
    totalFields_std
    staticFields_mean
    staticFields_min
    staticFields_max
    staticFields_std
    publicFields_mean
    publicFields_min
    publicFields_max
    publicFields_std
    privateFields_mean
    privateFields_min
    privateFields_max
    privateFields_std
    protectedFields_mean
    protectedFields_min
    protectedFields_max
    protectedFields_std
    defaultFields_mean
    defaultFields_min
    defaultFields_max
    defaultFields_std
    finalFields_mean
    finalFields_min
    finalFields_max
    finalFields_std
    synchronizedFields_mean
    synchronizedFields_min
    synchronizedFields_max
    synchronizedFields_std
    nosi_mean
    nosi_min
    nosi_max
    nosi_std
    loc_mean
    loc_min
    loc_max
    loc_std
    returnQty_mean
    returnQty_min
    returnQty_max
    returnQty_std
    loopQty_mean
    loopQty_min
    loopQty_max
    loopQty_std
    comparisonsQty_mean
    comparisonsQty_min
    comparisonsQty_max
    comparisonsQty_std
    tryCatchQty_mean
    tryCatchQty_min
    tryCatchQty_max
    tryCatchQty_std
    parenthesizedExpsQty_mean
    parenthesizedExpsQty_min
    parenthesizedExpsQty_max
    parenthesizedExpsQty_std
    stringLiteralsQty_mean
    stringLiteralsQty_min
    stringLiteralsQty_max
    stringLiteralsQty_std
    numbersQty_mean
    numbersQty_min
    numbersQty_max
    numbersQty_std
    assignmentsQty_mean
    assignmentsQty_min
    assignmentsQty_max
    assignmentsQty_std
    mathOperationsQty_mean
    mathOperationsQty_min
    mathOperationsQty_max
    mathOperationsQty_std
    variablesQty_mean
    variablesQty_min
    variablesQty_max
    variablesQty_std
    maxNestedBlocks_mean
    maxNestedBlocks_min
    maxNestedBlocks_max
    maxNestedBlocks_std
    anonymousClassesQty_mean
    anonymousClassesQty_min
    anonymousClassesQty_max
    anonymousClassesQty_std
    subClassesQty_mean
    subClassesQty_min
    subClassesQty_max
    subClassesQty_std
    lambdasQty_mean
    lambdasQty_min
    lambdasQty_max
    lambdasQty_std
    uniqueWordsQty_mean
    uniqueWordsQty_min
    uniqueWordsQty_max
    uniqueWordsQty_std
    modifiers_mean
    modifiers_min
    modifiers_max
    modifiers_std
    num_dependency_mean
    num_dependency_min
    num_dependency_max
    num_dependency_std
    num_line_affected_mean
    num_line_affected_min
    num_line_affected_max
    num_line_affected_std
    actual_num_of_classes_touched_mean
    actual_num_of_classes_touched_min
    actual_num_of_classes_touched_max
    actual_num_of_classes_touched_std
    commit_line_changed
    refactoring_perc
    refactoring_perc_time_taken
    cocomoII_time_taken
    geneticP_time_taken
    softwareMaintenance_time_taken
    fuzzy_time_taken
    

## 5. Building and Training Models 

**REARRANGE 04 - Model Building.ipynb**

The main function of this notebooks is to build a maching learning model using H2O AutoML using the following as features to predict ``refactoring_perc_time_taken_log``.

'cbo_mean','cbo_min','cbo_max', 'cbo_std',
 'wmc_mean','wmc_min','wmc_max', 'wmc_std',
 'dit_mean','dit_min','dit_max', 'dit_std',
 'rfc_mean', 'rfc_min', 'rfc_max', 'rfc_std',
 'lcom_mean', 'lcom_min', 'lcom_max', 'lcom_std',
 'totalMethods_mean', 'totalMethods_min', 'totalMethods_max', 'totalMethods_std',
 'staticMethods_mean', 'staticMethods_min', 'staticMethods_max', 'staticMethods_std',
 'publicMethods_mean', 'publicMethods_min', 'publicMethods_max', 'publicMethods_std',
 'privateMethods_mean', 'privateMethods_min', 'privateMethods_max', 'privateMethods_std',
 'protectedMethods_mean', 'protectedMethods_min', 'protectedMethods_max', 'protectedMethods_std',
 'defaultMethods_mean', 'defaultMethods_min', 'defaultMethods_max', 'defaultMethods_std',
 'abstractMethods_mean', 'abstractMethods_min', 'abstractMethods_max', 'abstractMethods_std',
 'finalMethods_mean', 'finalMethods_min', 'finalMethods_max', 'finalMethods_std',
 'synchronizedMethods_mean', 'synchronizedMethods_min', 'synchronizedMethods_max', 'synchronizedMethods_std',
 'totalFields_mean', 'totalFields_min', 'totalFields_max', 'totalFields_std',
 'staticFields_mean', 'staticFields_min', 'staticFields_max', 'staticFields_std',
 'publicFields_mean', 'publicFields_min', 'publicFields_max', 'publicFields_std',
 'privateFields_mean', 'privateFields_min', 'privateFields_max', 'privateFields_std',
 'protectedFields_mean', 'protectedFields_min', 'protectedFields_max', 'protectedFields_std',
 'defaultFields_mean', 'defaultFields_min', 'defaultFields_max', 'defaultFields_std',
 'finalFields_mean', 'finalFields_min', 'finalFields_max', 'finalFields_std',
 'synchronizedFields_mean', 'synchronizedFields_min', 'synchronizedFields_max', 'synchronizedFields_std',
 'nosi_mean', 'nosi_min', 'nosi_max', 'nosi_std',
 'loc_mean', 'loc_min', 'loc_max','loc_std',
 'returnQty_mean', 'returnQty_min', 'returnQty_max', 'returnQty_std',
 'loopQty_mean', 'loopQty_min', 'loopQty_max', 'loopQty_std',
 'comparisonsQty_mean', 'comparisonsQty_min', 'comparisonsQty_max', 'comparisonsQty_std',
 'tryCatchQty_mean','tryCatchQty_min', 'tryCatchQty_max', 'tryCatchQty_std',
 'parenthesizedExpsQty_mean','parenthesizedExpsQty_min', 'parenthesizedExpsQty_max', 'parenthesizedExpsQty_std',
 'stringLiteralsQty_mean', 'stringLiteralsQty_min', 'stringLiteralsQty_max', 'stringLiteralsQty_std',
 'numbersQty_mean', 'numbersQty_min', 'numbersQty_max', 'numbersQty_std',
 'assignmentsQty_mean', 'assignmentsQty_min', 'assignmentsQty_max', 'assignmentsQty_std',
 'mathOperationsQty_mean', 'mathOperationsQty_min', 'mathOperationsQty_max', 'mathOperationsQty_std',
 'variablesQty_mean', 'variablesQty_min', 'variablesQty_max', 'variablesQty_std',
 'maxNestedBlocks_mean', 'maxNestedBlocks_min', 'maxNestedBlocks_max', 'maxNestedBlocks_std',
 'anonymousClassesQty_mean', 'anonymousClassesQty_min', 'anonymousClassesQty_max', 'anonymousClassesQty_std',
 'subClassesQty_mean', 'subClassesQty_min', 'subClassesQty_max', 'subClassesQty_std',
 'lambdasQty_mean', 'lambdasQty_min', 'lambdasQty_max', 'lambdasQty_std',
 'uniqueWordsQty_mean', 'uniqueWordsQty_min', 'uniqueWordsQty_max', 'uniqueWordsQty_std',
 'modifiers_mean', 'modifiers_min', 'modifiers_max', 'modifiers_std',
 'num_dependency_mean', 'num_line_affected_mean'


**Inputs**:
1. Software Features (Predictors)
2. refactoring_perc_time_taken_log (Target Variable)

**Outputs**:
1. Maching Learning Model (H2O Automl)
    
Details of the model is given below.


```python
import h2o
h2o.init()
model_path = "models/EffortEstimationModelv3/Log_Regression_GBM_grid__1_AutoML_20220228_154246_model_3"
model = h2o.load_model(model_path)
model
```

    Checking whether there is an H2O instance running at http://localhost:54321 . connected.
    Warning: Your H2O cluster version is too old (11 months and 3 days)! Please download and install the latest version from http://h2o.ai/download/
    


<div style="overflow:auto"><table style="width:50%"><tr><td>H2O_cluster_uptime:</td>
<td>15 secs</td></tr>
<tr><td>H2O_cluster_timezone:</td>
<td>Asia/Kuala_Lumpur</td></tr>
<tr><td>H2O_data_parsing_timezone:</td>
<td>UTC</td></tr>
<tr><td>H2O_cluster_version:</td>
<td>3.32.1.7</td></tr>
<tr><td>H2O_cluster_version_age:</td>
<td>11 months and 3 days !!!</td></tr>
<tr><td>H2O_cluster_name:</td>
<td>H2O_from_python_tanji_1qzstq</td></tr>
<tr><td>H2O_cluster_total_nodes:</td>
<td>1</td></tr>
<tr><td>H2O_cluster_free_memory:</td>
<td>7.984 Gb</td></tr>
<tr><td>H2O_cluster_total_cores:</td>
<td>12</td></tr>
<tr><td>H2O_cluster_allowed_cores:</td>
<td>12</td></tr>
<tr><td>H2O_cluster_status:</td>
<td>locked, healthy</td></tr>
<tr><td>H2O_connection_url:</td>
<td>http://localhost:54321</td></tr>
<tr><td>H2O_connection_proxy:</td>
<td>{"http": null, "https": null}</td></tr>
<tr><td>H2O_internal_security:</td>
<td>False</td></tr>
<tr><td>H2O_API_Extensions:</td>
<td>Amazon S3, Algos, AutoML, Core V3, TargetEncoder, Core V4</td></tr>
<tr><td>Python_version:</td>
<td>3.7.3 final</td></tr></table></div>


    Model Details
    =============
    H2OGradientBoostingEstimator :  Gradient Boosting Machine
    Model Key:  GBM_grid__1_AutoML_20220228_154246_model_3
    
    
    Model Summary: 
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>number_of_trees</th>
      <th>number_of_internal_trees</th>
      <th>model_size_in_bytes</th>
      <th>min_depth</th>
      <th>max_depth</th>
      <th>mean_depth</th>
      <th>min_leaves</th>
      <th>max_leaves</th>
      <th>mean_leaves</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td></td>
      <td>33.0</td>
      <td>33.0</td>
      <td>3454.0</td>
      <td>2.0</td>
      <td>3.0</td>
      <td>2.575757</td>
      <td>3.0</td>
      <td>4.0</td>
      <td>3.666667</td>
    </tr>
  </tbody>
</table>
</div>


    
    
    ModelMetricsRegression: gbm
    ** Reported on train data. **
    
    MSE: 3.2044038310962666
    RMSE: 1.7900848670094573
    MAE: 1.424412915308594
    RMSLE: NaN
    Mean Residual Deviance: 3.2044038310962666
    
    ModelMetricsRegression: gbm
    ** Reported on cross-validation data. **
    
    MSE: 3.8037588457541194
    RMSE: 1.9503227542522594
    MAE: 1.5519355816853435
    RMSLE: NaN
    Mean Residual Deviance: 3.8037588457541194
    
    Cross-Validation Metrics Summary: 
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>mean</th>
      <th>sd</th>
      <th>cv_1_valid</th>
      <th>cv_2_valid</th>
      <th>cv_3_valid</th>
      <th>cv_4_valid</th>
      <th>cv_5_valid</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>mae</td>
      <td>1.5520502</td>
      <td>0.08237011</td>
      <td>1.4910178</td>
      <td>1.467426</td>
      <td>1.6308649</td>
      <td>1.5229915</td>
      <td>1.647951</td>
    </tr>
    <tr>
      <th>1</th>
      <td>mean_residual_deviance</td>
      <td>3.80427</td>
      <td>0.3442388</td>
      <td>3.4929738</td>
      <td>3.5808723</td>
      <td>4.040876</td>
      <td>3.6165812</td>
      <td>4.2900476</td>
    </tr>
    <tr>
      <th>2</th>
      <td>mse</td>
      <td>3.80427</td>
      <td>0.3442388</td>
      <td>3.4929738</td>
      <td>3.5808723</td>
      <td>4.040876</td>
      <td>3.6165812</td>
      <td>4.2900476</td>
    </tr>
    <tr>
      <th>3</th>
      <td>r2</td>
      <td>0.17276023</td>
      <td>0.03720194</td>
      <td>0.19935311</td>
      <td>0.19155143</td>
      <td>0.121441506</td>
      <td>0.20593041</td>
      <td>0.14552468</td>
    </tr>
    <tr>
      <th>4</th>
      <td>residual_deviance</td>
      <td>3.80427</td>
      <td>0.3442388</td>
      <td>3.4929738</td>
      <td>3.5808723</td>
      <td>4.040876</td>
      <td>3.6165812</td>
      <td>4.2900476</td>
    </tr>
    <tr>
      <th>5</th>
      <td>rmse</td>
      <td>1.9488872</td>
      <td>0.08738271</td>
      <td>1.8689499</td>
      <td>1.8923193</td>
      <td>2.0101929</td>
      <td>1.9017311</td>
      <td>2.071243</td>
    </tr>
    <tr>
      <th>6</th>
      <td>rmsle</td>
      <td>NaN</td>
      <td>0.0</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


    
    Scoring History: 
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th></th>
      <th>timestamp</th>
      <th>duration</th>
      <th>number_of_trees</th>
      <th>training_rmse</th>
      <th>training_mae</th>
      <th>training_deviance</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td></td>
      <td>2022-02-28 15:43:08</td>
      <td>1.653 sec</td>
      <td>0.0</td>
      <td>2.146777</td>
      <td>1.667285</td>
      <td>4.608652</td>
    </tr>
    <tr>
      <th>1</th>
      <td></td>
      <td>2022-02-28 15:43:08</td>
      <td>1.673 sec</td>
      <td>5.0</td>
      <td>2.014709</td>
      <td>1.580553</td>
      <td>4.059051</td>
    </tr>
    <tr>
      <th>2</th>
      <td></td>
      <td>2022-02-28 15:43:08</td>
      <td>1.693 sec</td>
      <td>10.0</td>
      <td>1.932624</td>
      <td>1.522236</td>
      <td>3.735036</td>
    </tr>
    <tr>
      <th>3</th>
      <td></td>
      <td>2022-02-28 15:43:08</td>
      <td>1.717 sec</td>
      <td>15.0</td>
      <td>1.881267</td>
      <td>1.490755</td>
      <td>3.539164</td>
    </tr>
    <tr>
      <th>4</th>
      <td></td>
      <td>2022-02-28 15:43:08</td>
      <td>1.742 sec</td>
      <td>20.0</td>
      <td>1.851262</td>
      <td>1.472042</td>
      <td>3.427170</td>
    </tr>
    <tr>
      <th>5</th>
      <td></td>
      <td>2022-02-28 15:43:08</td>
      <td>1.771 sec</td>
      <td>25.0</td>
      <td>1.823754</td>
      <td>1.451458</td>
      <td>3.326080</td>
    </tr>
    <tr>
      <th>6</th>
      <td></td>
      <td>2022-02-28 15:43:08</td>
      <td>1.792 sec</td>
      <td>30.0</td>
      <td>1.801777</td>
      <td>1.435145</td>
      <td>3.246400</td>
    </tr>
    <tr>
      <th>7</th>
      <td></td>
      <td>2022-02-28 15:43:08</td>
      <td>1.805 sec</td>
      <td>33.0</td>
      <td>1.790085</td>
      <td>1.424413</td>
      <td>3.204404</td>
    </tr>
  </tbody>
</table>
</div>


    
    Variable Importances: 
    


<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>variable</th>
      <th>relative_importance</th>
      <th>scaled_importance</th>
      <th>percentage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>uniqueWordsQty_min</td>
      <td>1156.941528</td>
      <td>1.000000</td>
      <td>0.268673</td>
    </tr>
    <tr>
      <th>1</th>
      <td>loc_min</td>
      <td>555.740234</td>
      <td>0.480353</td>
      <td>0.129058</td>
    </tr>
    <tr>
      <th>2</th>
      <td>cbo_std</td>
      <td>502.424957</td>
      <td>0.434270</td>
      <td>0.116676</td>
    </tr>
    <tr>
      <th>3</th>
      <td>privateFields_std</td>
      <td>247.563660</td>
      <td>0.213981</td>
      <td>0.057491</td>
    </tr>
    <tr>
      <th>4</th>
      <td>finalFields_std</td>
      <td>219.819809</td>
      <td>0.190001</td>
      <td>0.051048</td>
    </tr>
    <tr>
      <th>5</th>
      <td>nosi_min</td>
      <td>205.712997</td>
      <td>0.177808</td>
      <td>0.047772</td>
    </tr>
    <tr>
      <th>6</th>
      <td>staticFields_mean</td>
      <td>192.080521</td>
      <td>0.166024</td>
      <td>0.044606</td>
    </tr>
    <tr>
      <th>7</th>
      <td>nosi_mean</td>
      <td>106.552544</td>
      <td>0.092098</td>
      <td>0.024744</td>
    </tr>
    <tr>
      <th>8</th>
      <td>subClassesQty_mean</td>
      <td>103.979286</td>
      <td>0.089874</td>
      <td>0.024147</td>
    </tr>
    <tr>
      <th>9</th>
      <td>finalFields_mean</td>
      <td>102.416138</td>
      <td>0.088523</td>
      <td>0.023784</td>
    </tr>
    <tr>
      <th>10</th>
      <td>publicMethods_mean</td>
      <td>93.361984</td>
      <td>0.080697</td>
      <td>0.021681</td>
    </tr>
    <tr>
      <th>11</th>
      <td>nosi_max</td>
      <td>69.754166</td>
      <td>0.060292</td>
      <td>0.016199</td>
    </tr>
    <tr>
      <th>12</th>
      <td>privateFields_mean</td>
      <td>61.719482</td>
      <td>0.053347</td>
      <td>0.014333</td>
    </tr>
    <tr>
      <th>13</th>
      <td>lambdasQty_mean</td>
      <td>59.594707</td>
      <td>0.051511</td>
      <td>0.013839</td>
    </tr>
    <tr>
      <th>14</th>
      <td>cbo_min</td>
      <td>57.528152</td>
      <td>0.049724</td>
      <td>0.013360</td>
    </tr>
    <tr>
      <th>15</th>
      <td>dit_std</td>
      <td>55.994667</td>
      <td>0.048399</td>
      <td>0.013003</td>
    </tr>
    <tr>
      <th>16</th>
      <td>cbo_max</td>
      <td>52.219261</td>
      <td>0.045136</td>
      <td>0.012127</td>
    </tr>
    <tr>
      <th>17</th>
      <td>wmc_min</td>
      <td>50.148849</td>
      <td>0.043346</td>
      <td>0.011646</td>
    </tr>
    <tr>
      <th>18</th>
      <td>uniqueWordsQty_max</td>
      <td>49.575752</td>
      <td>0.042851</td>
      <td>0.011513</td>
    </tr>
    <tr>
      <th>19</th>
      <td>rfc_min</td>
      <td>46.571686</td>
      <td>0.040254</td>
      <td>0.010815</td>
    </tr>
  </tbody>
</table>
</div>


    
    See the whole table with table.as_data_frame()
    




    



## 6. Model Validation & 7. Estimation Techniques

**REARRANGE 05 - Model Validation.ipynb**

The main function of this notebooks is to 
1. Validate the Machine Learning Model built in the previous notebook as a baseline model.
    * MAE
    * SA
    * RE*
2. Compare the performance of the model against other software estimation models.
    * COCOMOII
    * GeneticP
    * SoftwareMaintenance
    * Fuzzy

Requirements of a baseline model.
1. Be simple to describe, implement, and interpret.
2. Be deterministic in its outcomes.
3. Be applicable to mixed qualitative and quantitative data.
4. Offer some explanatory information regarding the prediction by representing generalised properties of the underlying data.
5. Have no parameters within the modelling process that require tuning.
6. Be publicly available via a reference implementation and associated environment for execution.
7. Generally be more accurate than a random guess or an estimate based purely on the distribution of the response variable.
8. Be robust to different data splits and validation methods.
9. Do not be expensive to apply.
10. Offer comparable performance to standard methods.

**Inputs**:
1. Training Data
2. Testing Data
3. Machine Learning Model

**Outputs**:
1. Validation Results


```python

```
