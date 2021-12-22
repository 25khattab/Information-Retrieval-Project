import os,numpy,math
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import pandas as pd
from sys import exit


nonStopWords = ['where','to','in']
stopWords = [word for word in stopwords.words() if not word in nonStopWords] 
stopWords.extend([',','\'','"',':','.','?','!','&','^','/','(',')','{','}','[',']',';','//','@','#','$','%','*','-','|','<','>','`','~'])

def make_token_files():
    for file in os.listdir("files"):
            file_path = os.path.abspath("files/"+file)
            with open(file_path, 'r',encoding='utf8') as in_file:
                with open(os.path.abspath("token_files/"+file),'w') as out_file:  #ATTENTION
                    for line in in_file:
                        t = word_tokenize(line)
                        tokens = [word.lower() for word in t]
                        tokens_without_sw = [word for word in tokens if not word in stopWords]
                        for word in tokens_without_sw:
                            out_file.write(word)
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

def compute_TF_WTF_DF_IDF(index):
    tf={}
    wtf={}
    df={}
    idf={}
    for word in index:
        tf[word]={}
        wtf[word]={}
        for file in os.listdir("token_files"):
            docID=os.path.basename(file).split('.')[0]
            df[word]=[len(index[word])]
            idf[word]=[numpy.log10(len(os.listdir("files"))/(df[word][0]))]            #change 10 to make it generic
            if docID in index[word].keys():
                tf[word][docID]=len(index[word][docID])
                wtf[word][docID]=numpy.log(tf[word][docID])+1
            else:
                tf[word][docID]=0
                wtf[word][docID]=0
    return tf,wtf,df,idf      

def compute_tfidf(wtf,idf):
    tfidf={}
    for word in tf:
        tfidf[word]={}
        for docID in tf[word]:
            tfidf[word][docID]=wtf[word][docID]*idf[word][0]
            
    return tfidf

def compute_norm(tfidf):
    sumdic={}
    for word in tfidf:
        for docID in tfidf[word]:
            if not docID in sumdic:
                sumdic[docID]=[pow(tfidf[word][docID],2)]
            else:
                sumdic[docID][0]=sumdic[docID][0]+pow(tfidf[word][docID],2)
    for docID in sumdic:
        sumdic[docID]=[math.sqrt(sumdic[docID][0])]
    ntfidf = {}
    for word in tfidf:
        ntfidf[word]={}
        for docID in tfidf[word]: 
            ntfidf[word][docID]= tfidf[word][docID]/sumdic[docID][0]
    return ntfidf,sumdic

def find_inter(word,matches,dic):
    ans={}
    for docid in matches:
        if not word in dic:
            return {}
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

def compute_COSINE_SIM(matched,qntfidf,nor,q):
    result={}
    for word in q:
        for docID in matched:
            if not docID in result:
                result[docID]=nor[word][docID]*qntfidf[word][docID]
            else:
                result[docID]=result[docID]+nor[word][docID]*qntfidf[word][docID]
    return result   

def compute_qntfidf(matched,tfidf,q):
    qntfidf={}
    for word in q:
        if not word in tfidf : continue
        if not word in qntfidf:
            qntfidf[word]={}
        for docID in matched:
            qntfidf[word][docID]=tfidf[word][docID]
    qsum={}
    for word in q:
        for docID in qntfidf[word]:
            if not docID in qsum:
                qsum[docID]=qntfidf[word][docID]*qntfidf[word][docID]
            else :
                qsum[docID]=qsum[docID]+(qntfidf[word][docID]*qntfidf[word][docID])
    for docID in qsum:
        qsum[docID]=math.sqrt(qsum[docID])
    for word in qntfidf:
        for docID in qntfidf[word]:
            qntfidf[word][docID]=qntfidf[word][docID]/qsum[docID]

    return qntfidf


def display_Rank(result):
    result = sorted(result.items(), key=lambda x: x[1])
    result=dict(result)
    print("Query Result based on cosine similarity")
    print("docID :",end="")
    for key in result.keys() :
        print ("{:>6}".format(key) ,end='')
    print()

def display(ret,label):
    df=ret
    if len(df.iloc[0])==1:
        new_header = df.iloc[0]
        df = df[1:]
        df.columns = new_header
    print(label)
    print(df)
    print()

make_token_files()

dic={}
dic=createPositionalIndex()
#display(pd.DataFrame.from_dict(dic).T,"dic : ")
tf,wtf,df,idf=compute_TF_WTF_DF_IDF(dic)
tfidf=compute_tfidf(tf,idf)
nor,sumdic=compute_norm(tfidf)

########################################################
#printing matrices
display(pd.DataFrame.from_dict(tf).T,"TF :")
display(pd.DataFrame.from_dict(wtf).T,"WTF :")
display(pd.DataFrame.from_dict(df).T,"DF :")
display(pd.DataFrame.from_dict(idf).T,"IDF :")
display(pd.DataFrame.from_dict(tfidf).T,"TF-IDF :")
display(pd.DataFrame.from_dict(sumdic).T,"Doc Length :")
display(pd.DataFrame.from_dict(nor).T,"Normalized Values :")


#antony brutus caeser

while True:
    q = input()
    if(q=='/exit'):
        exit()
    q=word_tokenize(q)
    q_without_sw = [word for word in q if not word in stopWords]
    matches = []
    for id,word in enumerate(q_without_sw):
        if id==0:
            if word in dic:
                matches=dic[word]
        else:
            matches=find_inter(word,matches,dic)

    
    if len(matches):
        qntfidf={}
        qntfidf=compute_qntfidf(matches,tfidf,q_without_sw)
        cos=compute_COSINE_SIM(matches,qntfidf,nor,q_without_sw)
        display(pd.DataFrame([cos]).T,"COSINE SIMILARITY :")
        display_Rank(cos)
    else:
        print("no match try again...")
    