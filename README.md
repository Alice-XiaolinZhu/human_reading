## 1. Introduction to code files
The code is based on Michael Hahn & Frank Keller' study [Modeling Human Reading with Neural Attention](https://arxiv.org/abs/1608.05604) in modeling English reading behaviors. 

The directory include python files as follows:  

train_autoencoder.py file for pre-training reader-decoder network;  
train_attention_basic.py for jointly training the attention module without extra conditions given;  
train_attention_preview.py for training the attention module, allowing model to take into account the context, LM loss and preview characters;  
train_attention_saccadic_length.py for training the attention module to predict saccadic length;  
train.py file for training all the variants with different variables;  
test_attention_basic.py for basic attention module to predict the fixation sequence;  
test_attention_preview.py for attention module variants to predict the fixation sequence;  
test_attention_saccadic_length.py for attention module to predict the saccadic length;  
test.py for testing all the models once.  

In addition, it provides .ipynb files to extract evaluation data, run baselines and Linear Mixed-effects Models (LMMs) experiment with R.  
These files include:  
read_bsc_final.ipynb for reading Beijing Sentence Corpus (BSC) for eye-movement evaluation data and development and testing sentences;  
evaluation_final.py for evaluating model performance by calculating perplexity, accuracy and F1-score;  
run_baseline_final.ipynb to run baselines involved in this study;  
analysis_wrap-up_final.ipynb for analysing first character effect and wrap-up effect in BSC dataset;  
analysis_human_¥&¥_prediction_includeFirstChar_final.ipynb to analyse LMM results before removing the first character;  
analysis_human_¥&¥_prediction_excludeFirstChar_final.ipynb to analyse LMM results after removing the first character.  

## 2. Introdction to dataset
Dataset involved in this study include:
[CLUECorpus2020](https://github.com/CLUEbenchmark/CLUECorpus2020) for model training;   
and [Beijing Sentence Corpus](https://osf.io/vr3k8) for evaluation;   
and [Wikipedia Dump](http://download.wikipedia.com/zhwiki) for pre-training embedding models.


## 3. Experimental set up
Experimental set up is outlined as follows:

3.1. R setup  
Install R: https://cran.r-project.org/bin/macosx/. 
Install RStudio: https://www.rstudio.com/products/rstudio/download/. 
Check library path: print(.libPaths()).   
Install packages:   
install.packages(‘usethis’),   
install.packages(‘devtools’):   
namespace 'rlang' 0.4.5 is being loaded, but >= 0.4.10 is required. 
->delete rlang package from /library installed by RStudio in terminal. 
Installation of package ‘devtools’ had non-zero exit status. 
->install manually: https://cran.r-project.org/web/packages/pkgload/index.html. 
install.packages(‘IRkernel’).   
devtools::install_github('IRkernel/IRkernel'). 
IRkernel::installspec():  
jupyter-client has to be installed but "jupyter kernelspec --version" exited with code 127.  
->use anaconda terminal to do this: R, ..., q(). 


3.2. CWE embedding set up:  
git clone https://github.com/Leonard-Xu/CWE.git. 

3.3. JWE embedding set up:  
git clone https://github.com/HKUST-KnowComp/JWE.git. 
cd JWE/src.  
make all. 
./jwe. 
./run.sh. 

3.4. Cw2vec embedding set up:  
git clone https://github.com/bamtercelboo/cw2vec.git. 
cd word2vec && cd build. 
cmake ..  
make (add #include <stdexcept> in ./build/src/include/args.h). 
cd ../bin (/embeddings/cw2vec/words2vec/bin). 
./word2vec substoke -input ../wiki_process.txt -infeature ../../Simplified_Chinese_Feature/sin_chinese_feature.txt -output substroke_out.txt -lr 0.025 -dim 200 -ws 5 -epoch 5 -minCount 10 -neg 5 -loss ns -minn 3 -maxn 18 -thread 8 -t 1e-4 -lrUpdateRate 100. 

3.5. GWE embedding set up:  
git clone https://github.com/ray1007/GWE.git. 
Insert #define _POSIX_C_SOURCE 200112L before include. 
  

