# Natural Language Processing: Sentiment Analysis

## Detecting Stress on Text Using Bag of Words

# Authors:
 
Jennifer Rodriguez Trujillo

# Research Question: 

Mental Health Advocacy has exponentially grown in the past decade. In efforts to increase outreach, machine learning models such as Bag of Words have been used to determine text sentiment. From user's comments to the post's users view, the text is analyzed in attempt to target users who may display signs of distress. Bag of Words is a machine learning algorithm often used in Natural Learning Process. By using the algorithm Bag of Words and training the model using Logistic Regression, I will be using "TfidfVectorizer" and "CountVectorizer" in order to determine which method produces optimal accuracy. In this research project, I will be using a web scrapped data from subreddits. Furthermore, this dataset personal annecdotes and commentary in relation to mental health struggles.


# Background: 

The ability to detect mental destress in social media posts is crucial in that any form of outreach can make a difference. Countless of mental health statistics exist online and many of it revolves around the idea of out reach. Mental health has not alwauys received the attention it requires, but even then, it still remain taboo in certain communities and geographic locations to talk about mental health. For these vulneable communities, or potentially even households, it is pertinent to reach out to them through other means. Although technology is not readily available to all, it encompasses a broad population of teenagers: the current predominant group. Many teenagers reserve to social media as a form of relief. With that said, NLP can be a powerful tool to use in order to target vulnerable users. Although the data being used only encompasses a subsample, it permits us to receive of glimpse at what an optimal NLP model that detects stress may present itself to be. 


# Project:
In this project, given subreddit texts in realtion to mental health  struggles, I will be examining and analyzing text sentiment. I will be building a model using the Bag of Words algorithm and training it through Logistic Regression. I will then be using "TfidfVectorizer" and "CountVectorizer" in order to determine which method produces optimal accuracy.

# Data:

The data I will be using consists of subreddits collected via web scrapping. The following is a list of column names the data consists of:

subreddit                   ptsdassistanceptsdrelationshipssurvivorsofabus...
post_id                     8601tu8lbrx99ch1zh7rorpp9p2gbc7tx7et7iphly5m3k...
sentence_range              (15, 20)(0, 5)(15, 20)[5, 10][0, 5](30, 35)[25...
text                        said felt way sugget go rest trigger ahead you...
id                                                                   39028174
label                                                                    1488
confidence                                                            2295.86
social_timestamp                                                4308387751949
social_karma                                                            51828
syntax_ari                                                              13294
lex_liwc_WC                                                            244057
lex_liwc_Analytic                                                      100014
lex_liwc_Clout                                                         116211
lex_liwc_Authentic                                                     190272
lex_liwc_Tone                                                         94869.1
lex_liwc_WPS                                                          51621.7
lex_liwc_Sixltr                                                       42167.4
lex_liwc_Dic                                                           262087
lex_liwc_function                                                      166304
lex_liwc_pronoun                                                      56100.7
lex_liwc_ppron                                                        39696.6
lex_liwc_i                                                              25730
lex_liwc_we                                                           2170.16
lex_liwc_you                                                          2457.06
lex_liwc_shehe                                                        7685.52
lex_liwc_they                                                         1653.61
lex_liwc_ipron                                                        16372.5
lex_liwc_article                                                        14013
lex_liwc_prep                                                         37906.2
lex_liwc_auxverb                                                      29249.3
lex_liwc_adverb                                                       17158.1
lex_liwc_conj                                                         21566.1
lex_liwc_negate                                                       6422.12
lex_liwc_verb                                                           55146
lex_liwc_adj                                                          12381.4
lex_liwc_compare                                                      6491.32
lex_liwc_interrog                                                     4564.12
lex_liwc_number                                                        4098.5
lex_liwc_quant                                                         6375.3
lex_liwc_affect                                                       17304.4
lex_liwc_posemo                                                       7656.53
lex_liwc_negemo                                                       9378.83
lex_liwc_anx                                                          2594.49
lex_liwc_anger                                                        2635.18
lex_liwc_sad                                                          1698.41
lex_liwc_social                                                       30728.5
lex_liwc_family                                                       2114.91
lex_liwc_friend                                                       1575.75
lex_liwc_female                                                       4474.43
lex_liwc_male                                                         5610.37
lex_liwc_cogproc                                                      38582.5
lex_liwc_insight                                                       8102.8
lex_liwc_cause                                                        4899.64
lex_liwc_discrep                                                      5526.11
lex_liwc_tentat                                                       9435.66
lex_liwc_certain                                                      4382.64
lex_liwc_differ                                                       11272.3
lex_liwc_percept                                                      6376.87
lex_liwc_see                                                           1626.5
lex_liwc_hear                                                         1741.48
lex_liwc_feel                                                         2588.72
lex_liwc_bio                                                           7409.1
lex_liwc_body                                                         2010.96
lex_liwc_health                                                       3538.42
lex_liwc_sexual                                                        607.64
lex_liwc_ingest                                                       1150.66
lex_liwc_drives                                                       22378.2
lex_liwc_affiliation                                                  7768.64
lex_liwc_achieve                                                      3896.04
lex_liwc_power                                                        6851.88
lex_liwc_reward                                                       3884.81
lex_liwc_risk                                                         2043.75
lex_liwc_focuspast                                                    14356.2
lex_liwc_focuspresent                                                 34975.6
lex_liwc_focusfuture                                                  3358.13
lex_liwc_relativ                                                      40453.5
lex_liwc_motion                                                       5688.48
lex_liwc_space                                                        17865.6
lex_liwc_time                                                         17593.7
lex_liwc_work                                                         5323.22
lex_liwc_leisure                                                      2228.21
lex_liwc_home                                                         1802.45
lex_liwc_money                                                        2231.31
lex_liwc_relig                                                         328.25
lex_liwc_death                                                         402.46
lex_liwc_informal                                                     2343.21
lex_liwc_swear                                                         699.52
lex_liwc_netspeak                                                      724.26
lex_liwc_assent                                                        338.04
lex_liwc_nonflu                                                        354.09
lex_liwc_filler                                                        141.49
lex_liwc_AllPunc                                                      48398.3
lex_liwc_Period                                                       17216.1
lex_liwc_Comma                                                        10141.1
lex_liwc_Colon                                                         620.92
lex_liwc_SemiC                                                         350.29
lex_liwc_QMark                                                        1473.11
lex_liwc_Exclam                                                        546.29
lex_liwc_Dash                                                         1381.54
lex_liwc_Quote                                                        1343.83
lex_liwc_Apostro                                                      9014.79
lex_liwc_Parenth                                                      2210.69
lex_liwc_OtherP                                                       4098.82
lex_dal_max_pleasantness                                              7937.78
lex_dal_max_activation                                                7676.62
lex_dal_max_imagery                                                    8367.6
lex_dal_min_pleasantness                                              3087.75
lex_dal_min_activation                                                3178.84
lex_dal_min_imagery                                                    2838.6
lex_dal_avg_activation                                                4889.19
lex_dal_avg_imagery                                                    4360.3
lex_dal_avg_pleasantness                                               5333.7
social_upvote_ratio                                                    2393.9
social_num_comments                                                     28234
syntax_fk_grade                                                       15463.8
sentiment                                                             115.621



# References:
Millin, Oliver T., Jason C. Furtado, and Jeffrey B. Basara. "Characteristics, Evolution, and Formation of Cold Air Outbreaks in the Great Plains of the United States." Journal of Climate (2022): 1-37.

Rolnick, David, et al. "Tackling climate change with machine learning." ACM Computing Surveys (CSUR) 55.2 (2022): 1-96.
