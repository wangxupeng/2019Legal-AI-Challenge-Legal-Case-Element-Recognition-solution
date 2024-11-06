2019 Legal AI Challenge: Legal Case Element Recognition solution.（2019年法研杯要素识别冠军方案）
![hua](./pic/hua.png)


中文介绍：https://zhuanlan.zhihu.com/p/628656798

Detailed comments have been written in the code, please pay special attention to the parts where data is read and where the label tokens are retrieved.


## __Task Introduction__
The main objective of this task is to automatically extract important factual descriptions from case descriptions, and classify them according to a case element system designed by domain experts. Essentially, it is a short-text multilabel classification problem.

The results of case element extraction can be used in practical business requirements in the judicial field, such as case summaries, explainable case recommendations, and related knowledge recommendations.

Specifically, given relevant paragraphs in judicial documents, the system needs to judge each sentence in the document and identify the key case elements.

This task involves three domains, including marriage and family, labor disputes, and loan contracts.

## __Data Introduction__
The dataset used for this task mainly comes from the publicly available legal documents on the 'China Judgments Online' website. Each training data consists of a segment of the case description from a legal document, where each sentence has been labeled with corresponding category tags (it should be noted that the number of category tags for each sentence is not fixed). For example:

{"labels": ["DV9", "DV1", "DV2"], "sentence": "The plaintiff Xie Chunyou claimed that the plaintiff and defendant divorced in March 2014 due to emotional discord as ruled by the Hengyang County People's Court, and ruled that their daughter Zhou Yin (born on September 8, 2001, now studying at Zhengyang Middle School, Xidu Town, Hengyang County) would be raised by the defendant."}; DV represents the divorce category.

{"labels": ["LB2", "LB3"], "sentence": "The plaintiff Zhang Qian filed a lawsuit with the court and requested: 1. To order the defendant to make up the plaintiff's salary of 7200 yuan and compensation of 2400 yuan."}; LB represents the labor category.

{"labels": ["LN2", "LN6", "LN4"], "sentence": "This court finds the facts as follows: The defendant's brothers holdings group limited company borrowed three loans from the Industrial and Commercial Bank of China Yongkang Branch, with a total principal of RMB 40 million, as follows:"}; LN represents the loan category.

#### Label contents are as follows:

```python
Divorce = 
["Children born during marriage",
"Child custody for children with limited capacity for civil conduct",
"Common property of husband and wife",
"Payment of child support",
"Division of real estate",
"Separation after marriage",
"Divorce for the second time",
"Monthly payment of child support",
"Granting of divorce",
"Common debt of husband and wife",
"Personal property before marriage",
"Statutory divorce",
"Failure to perform family obligations",
"Existence of illegitimate child",
"Proper assistance",
"Failure to fulfill divorce agreement",
"Compensation for damages",
"Living separately for more than two years due to emotional discord",
"Children living with non-custodial parent",
"Personal property after marriage"]

Labor = 
["Termination of Labor Relationship",
"Payment of Wages",
"Payment of Economic Compensation",
"Unpaid Full Labor Remuneration",
"Existence of Labor Relationship",
"No Labor Contract Signed",
"Signed Labor Contract",
"Payment of Overtime Pay",
"Payment of Double Wages Compensation for No Signed Labor Contract",
"Payment of Work-Related Injury Compensation",
"No Labor Arbitration Filed",
"Refusal to Pay Illegal Termination Compensation",
"Economic Layoff",
"No Payment of Bonuses",
"Illegally Collecting Property from Employees",
"Special Job Categories",
"Payment of Subsidies for Work Injury, Funeral or Bereavement",
"Advance Notification of Termination by Employer",
"Legal Person Status Ceased",
"Agreement Reached through Mediation"]

Loan = 
["Transfer of creditor's rights",
"Loan amount of x thousand yuan",
"Loan certificate available",
"Lender is a financial institution",
"Repayment of loan",
"Loan borrowed by company|unit|other organization",
"Joint guarantee",
"Reminder for repayment",
"Payment of interest",
"Establishment of guarantee contract",
"Written promise of repayment available",
"Invalid|revoked|terminated guarantee contract",
"Refusal to perform repayment",
"Release of guarantor's guarantee responsibility",
"Guarantor not assuming guarantee responsibility",
"Pledgor is a company",
"Lender fails to provide loan as agreed in terms of date or amount",
"Loan borrowed by multiple individuals",
"Transfer of debt by debtor",
"Unclear agreed interest rate"]
```

#### Split multi-labels into single labels and then count:
![data_distribution.png](./pic/data_distribution.png "")


We can see that the marriage and family category accounts for 22% of the elements. Labor disputes category accounts for 11% of the elements. Loan contracts category accounts for 15% of the elements. Most of the sentences do not have relevant element information.

#### The distribution of data for the top 20 categories of marriage and family:
![top20.jpg](./pic/top20.jpg "")

#### The distribution of data for the top 20 categories of labor disputes:
![label_top20.jpg](./pic/label_top20.jpg "")

#### The distribution of data for the top 20 categories of loan contracts:
![loan_top20.jpg](./pic/loan_top20.jpg "")

From the above three graphs, it can be seen that __the data follows a severe long-tailed distribution__. After reviewing numerous papers, we found that label embedding can effectively solve this problem.

After reviewing many papers, we finally found inspiration from label embeddings[1] and achieved good results.

Next, let's take a look at the model's input and labels.

In the model input, we not only input the text but also add the corresponding category labels into Bert[2] (the final model is XLnet[3]), as follows:
```python
[CLS] Plaintiff Xie Chunyou alleges that the plaintiff and defendant divorced due to emotional discord in March 2014, and the court of Hengyang County ruled that the girl Zhou Yin (born on September 8, 2001, now studying at Zhengyang Middle School in Xidu Town, Hengyang County) born of the marriage should be raised by the defendant. [SEP] (Having Children After Marriage)...(Personal Property After Marriage)[SEP]
```
Labels: One-hot encoding is performed on different labels.

#### Model structure
![model_structure.png](./pic/model_structure.png "")
![model_structure2.png](./pic/model_structure2.png "")

The following is a comparison between the results with and without the use of label embedding.
![plus_label_embedding.jpg](./pic/plus_label_embedding.jpg "")
As we can see, the addition of label embedding has significantly improved the performance.

Here is our award-winning photo, LOL.
![photo.jpg](./pic/photo.jpg "")
Thanks to the award we received, I caught the attention of the CEO and CTO of Nanjing Aerospace Smart Technology Co., Ltd. Eventually, I was invited to join the Aerospace Science and Technology AI Laboratory. Looking back, it was an amazing experience.

---
__Reference__:
[1]Wang, S., Jiang, Z., Zheng, H.-T., & Li, B. (2018). Joint Embedding of Words and Labels for Text Classification. arXiv. arXiv:1805.04174.

[2]Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2018). Bert: Pre-training of deep bidirectional transformers for language understanding. arXiv preprint arXiv:1810.04805.

[3]Yang, Z., Dai, Z., Yang, Y., Carbonell, J. G., Salakhutdinov, R. R., & Le, Q. V. (2019). XLNet: Generalized Autoregressive Pretraining for Language Understanding. arXiv preprint arXiv:1906.08237.
