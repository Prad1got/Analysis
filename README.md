# Sentence Transformer Models Analysis

This work consists of performing an analysis on three Sentence Transformer Models, namely, all-MiniLM-L6-v2, all-mpnet-base-v2 and all-distilroberta-v1, 
to ascertain which one of these models is best suited for the Natural Language Processing (NLP) task of Question Answering (QA). The results and analysis
have been published as a Copyright work, owned by KJ Somaiya School of Engineering, Somaiya Vidyavihar University.

# A note on ```requirements.py```

Most projects come with a ```requirements.txt``` file. However, we have decided to write a Python file for the same, named ```requirements.py```. The intention
behind this is that while ```requirements.txt``` can just install files one-by-one as listed, it cannot guide the user around errors. However, in case of errors,
```requirements.py``` can give guides to the user as to what corrective steps can be taken to ensure that the installation is successful. In addition, while the
process of installation using ```requirements.txt``` does not print the different statements in different colors in the terminal, ```requirements.py``` does that.
This makes it easier for the user to track the installation process.

# Requirements

1. The machine intended to be used to run this project needs to have 8GB RAM (of which at least 4GB must be free; this implies that an 8GB RAM laptop should run 
Windows 10 or earlier as Windows 11 takes up 6 GB RAM). It is recommended that this project be run on a Windows environment as some file references and imports
in the code are formatted in Windows style.
2. The device needs to have Java installed, in order for CoreNLP to work.
3. Other requirements as specified by ```requirements.py``` while running it.

# Running Instructions
1. First, this repository should be cloned.
2. Next, the ```requirements.py``` file is run.
3. Next, any of these files, namely, ```sbertonly_sci-310.py```, ```sci4_reorder_edge_cosine-310.py``` and ```sci4_reorder_edge_plus_sbert-310.py``` can be run
   using any of the three Text Summarization Models as mentioned earlier. Note that the particular model to be used needs to be set in the code.

# Authors

1. Dr. Manish M Potey (Dean, KJ Somaiya School of Engineering)
2. Prof. Pradnya S Gotmare (Assistant Professor, Department of Computer Engineering, KJ Somaiya School of Engineering)
3. Mr. Sushant M Nair (Student Intern, LY BTech Computer Engineering, KJ Somaiya School of Engineering)
