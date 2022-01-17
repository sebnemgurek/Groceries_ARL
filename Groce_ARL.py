# Association Rules

#Kütüphaneleri import etme (Importing libraries)
import pandas as pd
from mlxtend.frequent_patterns import apriori, association_rules

# Loading dataset
df_ = pd.read_csv("Groceries_dataset.csv")
df = df_.copy()
df.head()


#Veri seti hakkında bilgiler (General information about the dataset)
df.shape
df.info()
df.isnull().sum()

#Tarih değişkenin tipini tarihe çevirme (Convert date variable type to date)
df["Date"]=pd.to_datetime(df["Date"])


#Ürünlerden kaçar tane satıldığı bilgisi (Information on how many of the products were sold)
df["itemDescription"].value_counts().head(10)

#En çok ürün satın alan 10 müşteri
df["Member_number"].value_counts().head(10)


#ARL Veri Yapısını Hazırlama (Preparing the ARL Data Structure)
dff=df.groupby(['Member_number', 'itemDescription'])['itemDescription'].count().unstack().fillna(0).applymap(
    lambda x: 1 if x > 0 else 0)

dff.iloc[0:5, 0:5]


# Birliktelik Kurallarının Çıkarılması (Removal of Association Rules)

# Tüm olası ürün birlikteliklerinin olasılıkları (Possibilities of all possible product combinations)
frequent_itemsets = apriori(dff, min_support=0.01, use_colnames=True)
frequent_itemsets.sort_values("support", ascending=False).head()


#Birliktelik kurallarının çıkarılması (Extraction of association rules)
rules = association_rules(frequent_itemsets, metric="lift", min_threshold=0.8)
rules.sort_values("support", ascending=False).head(10)



#antecedents   consequents     antecedent support  consequent support   support   confidence    lift     leverage    conviction
# (soda)       (whole milk)       0.313494            0.458184         0.151103    0.481997    1.051973    0.007465    1.045971

#Yorum
# 'Yogurt' ve "Whole milk" ürünleri alışverişlereien 0.15 inde birlikte görülmektedir.
# 'Yogurt' alan müşterilerin 0.53 ünün "Whole milk" de satın almıştır.
# 'Yogurt' olan sepetlerde "Whole milk" ürünün satışı 1.16 kat artmaktadır.

#Comment
# 'Yogurt' and 'Whole milk' products are seen together in 0.15 of the shopping.
# 0.53 of customers who bought 'yogurt' also bought 'Whole milk'.
# The sales of "Whole milk" product increases 1.16 times in baskets with 'yogurt'.
