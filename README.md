# OntoM

### The Ontology of Monotheism (OntoM) 
is a doctoral thesis developed by Bobiquins Estêvão de Mello at Graduate Program in Knowledge Engineering and Management in the Federal University of Santa Catarina, Florianópolis, SC, Brazil. 

You can find the full academic document "Tese_OntoM_BEM.pdf" in http://btd.egc.ufsc.br/

The structure and documentation of OntoM is accessible in the official site https://ontom.org/

The file Thesis-OntoM-BEM-v1.13.pdf is the official PhD thesis.
The file OntoM-v1.13.rtf contains the full documentation of classes and relations from the ontology.
The file OntoM-v1.13.asta is the editable archive generated by Astah Professional 8.4.1, release 2021/06/30; model version 39.

Please, cite this thesis as follow: 
### (MELLO, B. E. Ontology of Monotheism, 2021) or (MELLO, Bobiquins Estêvão de. Ontology of Monotheism, 2021).

Full quote:
Mello, Bobiquins Estêvão de.
Ontology of Monotheism (Bobiquins Estêvão de Mello);
advisor: Francisco Antonio Pereira Fialho,
joint advisor: José Leomar Todesco, 2021.
261 p.
Thesis (PhD) - The Federal University of Santa Catarina, Technological Center, Graduate Program of Knowledge Engineering and Management , Florianópolis, SC, Brazil, 2021.


### Repo for the Ontology of Monotheism experiments,
by Bobiquins Estevao de Mello and Bernardo Pereira Nunes:

To reproduce the results presented in the experiments from the thesis OntoM, you must follow the steps below. We tried to make it as simple as possible so that researchers not very familiar with python and BERT could reproduce the results with little effort.

1. Install Bert as a Service (follow the detailed instructions at https://github.com/hanxiao/bert-as-service)
2. Download the uncased large BERT pre-trained model at https://storage.googleapis.com/bert_models/2019_05_30/wwm_uncased_L-24_H-1024_A-16.zip
3. Run the Bert as a Serice pointing to the the unzipped downloaded model using the following command: bert-serving-start -model_dir /your_directory/wwm_uncased_L-24_H-1024_A-16 -num_worker=4 -port XXXX -max_seq_len NONE (we refer to the bert as a service link to understand more about the parameters used. Replace XXXX to any port available in your computer, we used port 8190).

### BERT is now ready to start generating the embeddings!

Now we need to run the scripts to generate the book and verse embeddings. To generate the book embeddings, we need to run the 'similarity_full.py' script.

###### run the script 'script/similarity_full.py "book_file_1.txt" "book_file_2.txt" "output_matrix_file.csv"

It will receive as parameters two text books (see raw_foundational_books folder) and output a file named "output_matrix_file.csv" with the similarity matrix of the inputted books. That is, it will return the dot product of the embeddings generated for each book. 

This matrix will be used to generate the heatmaps using the script 'religionProject.ipynb' located at the 'notebook_analysis' folder (more information below).

The generation of the matrix between two books is very expensive and generates a large file (for the foundational books used in the Web and Religion paper, the matrix file can reach approx. 1.75GB. Be patient in this step. Unfortunately, due to GitLab file size restrictions, we were unable to make these files available. However, all scripts are available to generate the matrices.

Cool! If you reach this point, you have everything running and the matrices generated.

Now, lets find the most similar verses from a book to another. For this, we will run the 'similarity.py' script.

###### run the script 'script/similarity.py "book_file_1.txt" "book_file_2.txt" "output_file.csv"

This command will generate the a CSV file containing the similarity score between the verse from book 1 and its most similar verse in the book 2. The CSV file format is: score, verse_book_1, most_similar_verse_book_2

PS: All the CSV files generated in this step can be found in the 'verse_similarity_per_book_pairs' folder as the size of the files are less than 100MB (max upload limit imposed by GitHub).

Note that you need to have the bert-as-service running to execute both scripts as they use it to generate the embeddings and output the CSV files.

This is all! You successfully generated all files to carry out the analysis using the notebook.

### Final step: Run the notebook script!

The notebook script can be found at the 'notebook_analysis' folder, download this file and run using the jupyter notebook (see how to install Jupyter at https://jupyter.org/install).

You might need to update the notebook script to point to the generated CSV files in the previous steps.

PS: A HTML version of the Notebook with the final results ('analysis_results.html') is also in the 'notebook_analysis' folder. Download the file to your computer and open it to see the results. If you only click on this, it will show the source code, which is useless for you. Download it and then open it as HTML page in your preferred browser.

### t-SNE visualization

To generate a 2D visualization of the embeddings (1024-d), we use a technique called t-SNE (read more at https://lvdmaaten.github.io/tsne/).

The first step is to generate the embeddings for each book running the following script:

###### run the script 'script/generateEmbeddings.py "book_file_1.txt" "output_embeddings_book_1_file.csv"

You must run this step for each foundational book. Again, the files are not available here due to the file size limit imposed by github. 

Finally, open the jupyter notebook and run the script 'tsne.ipynb' to generate the 2D visualization for the high-dimensional vectors generated by BERT. You may need to update the file location for the embeddings in the notebook.

### DONE! Now you can run additional experiments, analysis and share with us!
