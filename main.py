import os,numpy,math
from nltk.corpus.reader.rte import norm
from re import match
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd

haha="?-/(){}[].,~!@#$%^&*\'\":"

def make_token_files():
    for file in os.listdir("files"):
            file_path = os.path.abspath("files/"+file)
            with open(file_path, 'r',encoding='utf8') as in_file:
                with open(os.path.abspath("token_files/"+file),'w') as out_file:  #ATTENTION
                    for line in in_file:
                        t = word_tokenize(line)
                        tokens_without_sw = [word for word in t if not word in stopwords.words()]
                        tokens_without_s = [word for word in tokens_without_sw if not word in haha ]
                        for word in tokens_without_s:
                            out_file.write(word.lower())
                            out_file.write("\n")

def createPositionalIndex():
    index = {}
    for file in os.listdir("token_files"):
        file_path = os.path.abspath("token_files/"+file)
        with open(file_path, 'r') as in_file:
            for idx, word in enumerate(in_file.read().split('\n'),start=1):
                docID=os.path.basename(file).split('.')[0]
                if len(word)==0:continue
                if not word in index:
                    index[word]={}
                    index[word][docID]= [idx]
                else: 
                    
                    if not docID in index[word]:
                        index[word][docID]=[idx]
                    else:
                        index[word][docID].append(idx)
    return index

def compute_TF_DF_IDF(index):
    tf={}
    df={}
    
    idf={}
    for word in index:
        tf[word]={}
        
        for file in os.listdir("token_files"):
            docID=os.path.basename(file).split('.')[0]
            df[word]=[len(index[word])]
            idf[word]=[numpy.log10(10/(df[word][0]+1))]            #change 10 to make it generic
            if docID in index[word].keys():
                tf[word][docID]=len(index[word][docID])
            else:
                tf[word][docID]=0
    return tf,df,idf      

def compute_tfidf(tf,idf):
    tfidf={}
    for word in tf:
        tfidf[word]={}
        for docID in tf[word]:
            tfidf[word][docID]=round((numpy.log(tf[word][docID]+1))*idf[word][0],2)
    return tfidf

def compute_norm(tfidf):
    sumdic=[]
    for word in tfidf:
        i=0
        for value in tfidf[word]:
            if len(sumdic)<=i:
                sumdic.append(value*value)
            else:
                sumdic[i]=sumdic[i]+(value*value)
            i=i+1
    i=0
    for docId in sumdic:
        sumdic[i]=math.sqrt(docId)
    ntfidf = tfidf
    for word in ntfidf:
        i=0
        for docId in ntfidf[word]:
            
            ntfidf[word][i]= ntfidf[word][i]/sumdic[i]
            i=i+1
    return ntfidf

def display(d,n):
    print(n+'\n')
    for k in d:
        print("{:<15}".format(k),end=" ")
        for x in d[k]:
            print("{:<10}".format(x),end=" ")
        print()
    print("\n")
def find_inter(word,matches,dic):
    ans={}
    for docid in matches:
        if docid in dic[word]:
            l1=matches[docid]
            l2=dic[word][docid]
            i1=0
            i2=0
            
            while i1<len(l1) and i2<len(l2):
                
                if l1[i1] == l2[i2]-1:
                    if not docid in ans:
                        ans[docid]=[l2[i2]]
                    else:
                        
                        ans[docid].append(l2[i2])
                    i1=i1+1
                    i2=i2+1
                elif l1[i1] < l2[i2]-1 :
                    i1=i1+1
                else :
                    i2=i2+1
    return ans


make_token_files()

dic={}
dic=createPositionalIndex()
tf,df,idf=compute_TF_DF_IDF(dic)

tfidf=compute_tfidf(tf,idf)

print(pd.DataFrame.from_dict(tf).T)
print()
print(pd.DataFrame.from_dict(df).T)
print()
print(pd.DataFrame.from_dict(idf).T)
print()
print(pd.DataFrame.from_dict(tfidf).T)
#nor=compute_norm(tfidf)
#print(nor)


#antony brutus caeser

# while True:
#     q = [a for a in input().lower().split(' ') if a != ""]  #remove stop wordssss
#     matches = []
#     if len(q) == 1 and q[0] == '/exit':
#         exit()
#     for id,word in enumerate(q):
#         if id==0:
#             matches=dic[word]
#         else:
#             matches=find_inter(word,matches,dic)
#     print("docID : ",end=" ")
#     for key, value in matches.items() :
#         print (key ,end=' ')
#     print()   