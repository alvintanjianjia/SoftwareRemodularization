# SoftwareRemodularization (E-SC4R & REARRANGE)

# 1. Introduction
Maintenance of existing software requires a large amount of time for comprehending the source code. The architecture of a software, however, may not be clear to maintainers if up-to-date documentation is not available. Software clustering is often used as a remodularisation and architecture recovery technique to help recover a semantic representation of the software design.

**Explaining Software Clustering for Remodularisation (E-SC4R)**, to evaluate the effectiveness of different software clustering approaches. The proposed approach provides a better understanding of the algorithms’ behaviour by showing a 2D representation of the effectiveness of clustering techniques. 

**REARRANGE: An Effort Estimation Approach for Software
Clustering-based Remodularisation (REARRANGE)** aims to provide developers with refactoring recommendations and an estimate of person-hours needed to convert the current source code to the recommended structure.

# 2. Notebooks
## REARRANGE
The following notebooks contains the full end to end pipeline.
* [REARRANGE 01 - Crawl Github Commits](REARRANGE%2001%20-%20Crawl%20Github%20Commits.ipynb)
* [REARRANGE 02 - Crawl Refactoring Miner Data](REARRANGE%2002%20-%20Crawl%20Refactoring%20Miner%20Data.ipynb)
* [REARRANGE 03 - Merge Data (Github Commits, Refactoring Miner, Depends, CKMetrics)](REARRANGE%2003%20-%20Merge%20Data%20%28Github%20Commit%2C%20Refactoring%20Miner%2C%20Depends%2C%20CKMetrics%29.ipynb)
* [REARRANGE 04 - Model Building](REARRANGE%2004%20-%20Model%20Building.ipynb)
* [REARRANGE 05 - Model Validation](REARRANGE%2005%20-%20Model%20Validation.ipynb)

# 3. Datasets
## REARRANGE
Due to the large size of the entire dataset, sample datasets are uploaded at the following links. The full dataset can be crawled from the notebooks provided above.

* [Sample Commit Raw Data](https://figshare.com/s/6dde27bcd34e661d40c2)
* [Sample Refactoring Raw Data](https://figshare.com/s/f832c4592175974a6ec4)
* [Processed Data](https://figshare.com/s/ae4ec5b827b1b43ea6ba)

# 4. How to Use
## Download and Installation
Packages & Software Required
- Python 3.6
- Java JDK-11.0.12

## To Run Front End Streamlit
streamlit run streamlit/src/app.py --global.dataFrameSerialization="legacy"



# 5. How to contribute
## Support new languages
MION so far only supports Java. Please feel free to leverage this framework to add your own SoftwareRemodularizator. The effort needed for each language varies a lot.

## Enhance language features and fix issues
Currently, the parsing of the source files are done at a basic level. There are many language-specific features that requires special attention and should be taken into consideration during the parsing of the source files.

## Create useful tools
You could use this project as a building block to create various tools, either open source or commercial, for productions or research, such as GUI tools, code visualization tools, etc.

## Become a sponsor
It will help us a great deal if your company or institue becomes a sponsor of our project. Your donation could help us to be independent, sustainable, and support more contributors.

## Tell us your thoughts
You are welcome to use this project in your projects, either open source or commercial ones, as well as software engineering research. Your feedback is highly appreciated. We will also be thankful if you could help us spread the words and encourage more people to use it.

## Acknowledgement
This project is built upon the excellent work of other researchers, which includes
* Depends (https://github.com/multilang-depends/depends#output)
* CK Metrics (https://github.com/mauricioaniche/ck)

## Authors
* Alvin Tan Jian Jia (https://github.com/alvintanjianjia)
* Dr Chun Chong Yong (https://github.com/cychong9228) 
