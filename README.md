# Natural Language Processing: Sentiment Analysis

## Detecting Stress on Text Using Bag of Words

# Authors:
 
Jennifer Rodriguez Trujillo

# Research Question: 

Mental Health Advocacy has exponentially grown in the past decade. In efforts to increase outreach, machine learning models such as Bag of Words have been used to determine text sentiment. From user's comments to the post's users view, the text is analyzed in attempt to target users who may display signs of distress. Bag of Words is a machine learning algorithm often used in Natural Learning Process. By using the algorithm Bag of Words and training the model using Logistic Regression, I will be using "TfidfVectorizer" and "CountVectorizer" in order to determine which method produces optimal accuracy. In this research project, I will be using a web scrapped data from subreddits. Furthermore, this dataset personal annecdotes and commentary in relation to mental health struggles.


# Background: 

The ability to detect mental destress in social media posts is crucial in that any form of outreach can make a difference. Currently, we see many forms of outreach being done in short genres,such as Twitter, but what is currently lacking is the ability for models to detect stress in larger domains. Countless of mental health statistics exist online and many of it revolves around the idea of out reach. Mental health has not always received the attention it requires, but even then, it still remain taboo in certain communities and geographic locations to talk about mental health. For these vulneable communities, or potentially even households, it is pertinent to reach out to them through other means. Although technology is not readily available to all, it encompasses a broad population of teenagers: the current predominant group. Many teenagers reserve to social media as a form of relief. With that said, NLP can be a powerful tool to use in order to target vulnerable users. Although the data being used only encompasses a subsample, it permits us to receive of glimpse at what an optimal NLP model that detects stress may present itself to be. 


# Project:
In this project, given subreddit texts in realtion to mental health  struggles, I will be examining and analyzing text sentiment. I will be building a model using the Bag of Words algorithm and training it through Logistic Regression. I will then be using "TfidfVectorizer" and "CountVectorizer" in order to determine which method produces optimal accuracy.

# Data:

The data consists of subreddits from five different categories(n = 190k )collected via web scrapping. The following is a list of column names the data consists of:

```
subreddit                   ptsdassistanceptsdrelationshipssurvivorsofabus... <br/>
post_id                     8601tu8lbrx99ch1zh7rorpp9p2gbc7tx7et7iphly5m3k... <br/>
sentence_range              (15, 20)(0, 5)(15, 20)[5, 10][0, 5](30, 35)[25... <br/>
text                        said felt way sugget go rest trigger ahead you... <br/>
id                                                                   39028174 <br/>
label                                                                    1488 <br/>
confidence                                                            2295.86 <br/>
social_timestamp                                                4308387751949 <br/>
social_karma                                                            51828 <br/>
syntax_ari                                                              13294 <br/>
lex_liwc_WC                                                            244057 <br/>
lex_liwc_Analytic                                                      100014 <br/>
lex_liwc_Clout                                                         116211 <br/>
lex_liwc_Authentic                                                     190272 <br/>
lex_liwc_Tone                                                         94869.1 <br/>
lex_liwc_WPS                                                          51621.7 <br/>
lex_liwc_Sixltr                                                       42167.4 <br/>
lex_liwc_Dic                                                           262087 <br/>
lex_liwc_function                                                      166304 <br/>
lex_liwc_pronoun                                                      56100.7 <br/>
lex_liwc_ppron                                                        39696.6 <br/>
lex_liwc_i                                                              25730 <br/>
lex_liwc_we                                                           2170.16 <br/>
lex_liwc_you                                                          2457.06 <br/>
lex_liwc_shehe                                                        7685.52 <br/>
lex_liwc_they                                                         1653.61 <br/>
lex_liwc_ipron                                                        16372.5 <br/>
lex_liwc_article                                                        14013 <br/>
lex_liwc_prep                                                         37906.2 <br/>
lex_liwc_auxverb                                                      29249.3 <br/>
lex_liwc_adverb                                                       17158.1 <br/>
lex_liwc_conj                                                         21566.1 <br/>
lex_liwc_negate                                                       6422.12 <br/>
lex_liwc_verb                                                           55146 <br/>
lex_liwc_adj                                                          12381.4 <br/>
lex_liwc_compare                                                      6491.32 <br/>
lex_liwc_interrog                                                     4564.12 <br/>
lex_liwc_number                                                        4098.5 <br/>
lex_liwc_quant                                                         6375.3 <br/>
lex_liwc_affect                                                       17304.4 <br/>
lex_liwc_posemo                                                       7656.53 <br/>
lex_liwc_negemo                                                       9378.83 <br/>
lex_liwc_anx                                                          2594.49 <br/>
lex_liwc_anger                                                        2635.18 <br/>
lex_liwc_sad                                                          1698.41 <br/>
lex_liwc_social                                                       30728.5 <br/>
lex_liwc_family                                                       2114.91 <br/>
lex_liwc_friend                                                       1575.75 <br/>
lex_liwc_female                                                       4474.43 <br/>
lex_liwc_male                                                         5610.37 <br/>
lex_liwc_cogproc                                                      38582.5 <br/>
lex_liwc_insight                                                       8102.8 <br/>
lex_liwc_cause                                                        4899.64 <br/>
lex_liwc_discrep                                                      5526.11 <br/>
lex_liwc_tentat                                                       9435.66 <br/>
lex_liwc_certain                                                      4382.64 <br/>
lex_liwc_differ                                                       11272.3 <br/>
lex_liwc_percept                                                      6376.87 <br/>
lex_liwc_see                                                           1626.5 <br/>
lex_liwc_hear                                                         1741.48 <br/>
lex_liwc_feel                                                         2588.72 <br/>
lex_liwc_bio                                                           7409.1 <br/>
lex_liwc_body                                                         2010.96 <br/>
lex_liwc_health                                                       3538.42 <br/>
lex_liwc_sexual                                                        607.64 <br/>
lex_liwc_ingest                                                       1150.66 <br/>
lex_liwc_drives                                                       22378.2 <br/>
lex_liwc_affiliation                                                  7768.64 <br/>
lex_liwc_achieve                                                      3896.04 <br/>
lex_liwc_power                                                        6851.88 <br/>
lex_liwc_reward                                                       3884.81 <br/>
lex_liwc_risk                                                         2043.75 <br/>
lex_liwc_focuspast                                                    14356.2 <br/>
lex_liwc_focuspresent                                                 34975.6 <br/>
lex_liwc_focusfuture                                                  3358.13 <br/>
lex_liwc_relativ                                                      40453.5 <br/>
lex_liwc_motion                                                       5688.48 <br/>
lex_liwc_space                                                        17865.6 <br/>
lex_liwc_time                                                         17593.7 <br/>
lex_liwc_work                                                         5323.22 <br/>
lex_liwc_leisure                                                      2228.21 <br/>
lex_liwc_home                                                         1802.45 <br/>
lex_liwc_money                                                        2231.31 <br/>
lex_liwc_relig                                                         328.25 <br/>
lex_liwc_death                                                         402.46 <br/>
lex_liwc_informal                                                     2343.21 <br/>
lex_liwc_swear                                                         699.52 <br/>
lex_liwc_netspeak                                                      724.26 <br/>
lex_liwc_assent                                                        338.04 <br/>
lex_liwc_nonflu                                                        354.09 <br/>
lex_liwc_filler                                                        141.49 <br/>
lex_liwc_AllPunc                                                      48398.3 <br/>
lex_liwc_Period                                                       17216.1 <br/>
lex_liwc_Comma                                                        10141.1 <br/>
lex_liwc_Colon                                                         620.92 <br/>
lex_liwc_SemiC                                                         350.29 <br/>
lex_liwc_QMark                                                        1473.11 <br/>
lex_liwc_Exclam                                                        546.29 <br/>
lex_liwc_Dash                                                         1381.54 <br/>
lex_liwc_Quote                                                        1343.83 <br/>
lex_liwc_Apostro                                                      9014.79 <br/>
lex_liwc_Parenth                                                      2210.69 <br/>
lex_liwc_OtherP                                                       4098.82 <br/>
lex_dal_max_pleasantness                                              7937.78 <br/>
lex_dal_max_activation                                                7676.62 <br/>
lex_dal_max_imagery                                                    8367.6 <br/>
lex_dal_min_pleasantness                                              3087.75 <br/>
lex_dal_min_activation                                                3178.84 <br/>
lex_dal_min_imagery                                                    2838.6 <br/>
lex_dal_avg_activation                                                4889.19 <br/>
lex_dal_avg_imagery                                                    4360.3 <br/>
lex_dal_avg_pleasantness                                               5333.7 <br/>
social_upvote_ratio                                                    2393.9 <br/>
social_num_comments                                                     28234 <br/>
syntax_fk_grade                                                       15463.8 <br/>
sentiment                                                             115.621 <br/>
```


# References:
Turcan, E., & McKeown, K. (2019). Dreaddit: A Reddit dataset for stress analysis in social media. arXiv preprint arXiv:1911.00133.

Rolnick, David, et al. "Tackling climate change with machine learning." ACM Computing Surveys (CSUR) 55.2 (2022): 1-96.
