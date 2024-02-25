# Legal Documents Data Science exercise

Purpose of this test is to evaluate your ability to produce a good code and analysis while discovering a few aspects of the Machine Learning Engineer's job at Doctrine.

## Setup
 - Run `./init-project.sh`

## Objective: automatically detect the legal domain of court decisions
#### Context
You are a member of the Search Squad, and you want to develop a « legal domain filter » for our caselaw search-engine. This domain filter will allow to restrict the search results to court decisions belonging to a given « legal domain ».

To do that, you have collected data available online about a given jurisdiction. You have focused on the four main legal domains (civil, criminal, commercial and social) and have stored these decisions in 4 `.csv` files in the [data folder](https://github.com/DoctrineLegal/data-technical-test/tree/master/data) of this repository. Each file contains between 100 and 10 000 decisions (HTML format or plain text format) and corresponds to a specific legal domain, for instance `CIV.csv` stands for « civil ».
#### Missions
In order to build this new feature, you need to train a classifier that will predict the legal domain of new decisions, given nothing but their textual content. You will focus on the 4 main legal domains provided in your dataset.

The decisions of the provided dataset use standardized templates and look the same (format, syntax, ...), think about how your can make your classifier work on more diverse data.

What do you think of this approach? What are its limits? (scalability, generalization given the origin of your dataset, ...)
