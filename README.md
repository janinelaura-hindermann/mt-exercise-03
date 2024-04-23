# MT Exercise 3: Pytorch RNN Language Models

This repo shows how to train neural language models using [Pytorch example code](https://github.com/pytorch/examples/tree/master/word_language_model). Thanks to Emma van den Bold, the original author of these scripts. 

# Requirements

- This only works on a Unix-like system, with bash.
- Python 3 must be installed on your system, i.e. the command `python3` must be available
- Make sure virtualenv is installed on your system. To install, e.g.

    `pip install virtualenv`

# Steps

Clone this repository in the desired place:

    git clone https://github.com/moritz-steiner/mt-exercise-03
    cd mt-exercise-03

Create a new virtualenv that uses Python 3. Please make sure to run this command outside of any virtual Python environment:

    ./scripts/make_virtualenv.sh

**Important**: Then activate the env by executing the `source` command that is output by the shell script above.

Download and install required software:

    ./scripts/install_packages.sh

Download and preprocess data:

    ./scripts/download_data.sh

Train a model:

    ./scripts/train.sh

The training process can be interrupted at any time, and the best checkpoint will always be saved.

Generate (sample) some text from a trained model with:

    ./scripts/generate.sh

-----

# Task 1

## Finding an interesting data set

For this first part, refer to the folder `Family-Guy-Dialogues`.

From [Kaggle](https://www.kaggle.com/datasets/eswarreddy12/family-guy-dialogues-with-various-lexicon-ratings), we downloaded the file `Family_Guy_Final_NRC_AFINN_BING.csv`, which contains dialogue data from all episodes spanning seasons 1 to 19 of the animated television series "Family Guy", along with associated ratings from various lexicons.
The `convert_cvs_to_txt.py` file extracts the column “Dialogues” from the .cvs file and writes it in a .txt file.
Since the original file (`attempts` > `family_guy_dialogues_without_filtering_lines.txt`) turned out to be very large (155242 lines/segments), we implemented a “filter” in the program that only writes in the .txt file dialogues that are at least 17 words long (>= 17).

The different attempts we made are available in the `attempts` folder. Below are reported the number of lines for each "filter" (minimum word number) attempt:

	>= 10 -> 39321 lines
	>= 15 -> 12632 lines
	>= 16 -> 10045 lines
	>= 17 -> 8043  lines
	>= 18 -> 6434  lines
	>= 20 -> 4239  lines

We also removed duplicate lines. This way our data set `family_guy_dialogues.txt` (also available in the `data` > `familyguy`> `raw`folder) has 7514 lines/segments.


## Preprocessing

We modified the `download_data.sh` script as fallows:
- removed the `# link default training data for easier access`part
- removed the `# download a different interesting data set!`part
- adapted the `# preprocess slightly` and the `# tokenize, fix vocabulary upper bound` parts to our new data set (we kept a vocabulary size of 5000)
- adapted the `# split into train, valid and test` part (see below)

After running `preprocess_raw.py` and `preprocess.py` we obtained the file `family_guy_dialogues.preprocessed.txt`of 9951 lines/segments.
	
 	wc -l data/familyguy/raw/family_guy_dialogues.preprocessed.txt
    # 9951 data/familyguy/raw/family_guy_dialogues.preprocessed.txt
    
    
Taking a closer look at the `grimm` example, we noticed that the splitting ratios are about 75% for training, 10% for validation, and 10% for testing.

	wc -l data/grimm/*.txt
    # 400 data/grimm/test.txt
    # 2955 data/grimm/train.txt
    # 400 data/grimm/valid.txt
    # 3755 total

The actual total is 3915 `data/grimm/raw/tales.preprocessed.txt`.

	
Therefore, for our dataset of 9951 lines:
- Validation: 10% of 9951 = 995
- Testing:    10% of 9951 = 995
- Training:   80% of 9951 = 7960
	
So we adapted the `# split into train, valid and test` part of the script as follows:
	
	validation: first 995 lines from the preprocessed text file
	# head -n 995 $data/familyguy/raw/family_guy_dialogues.preprocessed.txt | tail -n 995 > $data/familyguy/family_guy_dialogues/valid.txt
	
	testing: lines from 996 to 1990 from the preprocessed text file
	# head -n 1990 $data/familyguy/raw/family_guy_dialogues.preprocessed.txt | tail -n 995 > $data/familyguy/family_guy_dialogues/test.txt
	
	training: all lines from the 1991st line to the end of the preprocessed text file
	# tail -n +1991 $data/familyguy/raw/family_guy_dialogues.preprocessed.txt > $data/familyguy/family_guy_dialogues/train.txt
	
...and when we checked:

	wc -l data/familyguy/*.txt                                    
    # 995 data/familyguy/test.txt
    # 7961 data/familyguy/train.txt
    # 995 data/familyguy/valid.txt
    # 9951 total


## Training a model and generating a sample text

At this point we ran the `train.sh`script (we changed `grimm` with `familyguy`) to train the model and afterwards the `generate.sh` script to generate some sample text.
First, we trained the model with the standard settings, and then we experimented a little by training the model with different parameters. The results are reported bellow.

### Standard settings
Parameters: epochs 40 \ log-interval 100 \ emsize 200 \ nhid 200 \ dropout 0.5

Standard settings - Training

	| epoch   1 |   100/  233 batches | lr 20.00 | ms/batch 62.01 | loss  6.28 | ppl   531.27
	| epoch   1 |   200/  233 batches | lr 20.00 | ms/batch 60.85 | loss  5.35 | ppl   209.92
	-----------------------------------------------------------------------------------------
	| end of epoch   1 | time: 14.96s | valid loss  5.12 | valid ppl   167.31
	-----------------------------------------------------------------------------------------
	
	[...]
	
	-----------------------------------------------------------------------------------------
	| epoch  20 |   100/  233 batches | lr 5.00 | ms/batch 62.14 | loss  4.06 | ppl    57.86
	| epoch  20 |   200/  233 batches | lr 5.00 | ms/batch 61.58 | loss  3.98 | ppl    53.50
	-----------------------------------------------------------------------------------------
	| end of epoch  20 | time: 15.07s | valid loss  4.38 | valid ppl    79.58
	-----------------------------------------------------------------------------------------
	
	[...]
	
	-----------------------------------------------------------------------------------------
	| epoch  40 |   100/  233 batches | lr 0.02 | ms/batch 62.07 | loss  3.99 | ppl    54.32
	| epoch  40 |   200/  233 batches | lr 0.02 | ms/batch 61.31 | loss  3.91 | ppl    49.77
	-----------------------------------------------------------------------------------------
	| end of epoch  40 | time: 15.01s | valid loss  4.36 | valid ppl    78.41
	-----------------------------------------------------------------------------------------
	=========================================================================================
	| End of training | test loss  4.28 | test ppl    72.03
	=========================================================================================
	time taken:
	604 seconds

Standard settings - Generated text

	. <eos> What &apos;s on from this point ? <eos> We put a panic PBS base on a <unk> dry
	<unk> and 00 base &apos; n &apos; cry He &apos;s no one for someone else , or or whatever a
	five sex . <eos> Well , you &apos;ve always asked me to bring us to <unk> in the <unk> of
	the rug and now I want you . <eos> On the school . <eos> Who you have said the or
	my daughters was just , like , it &apos;s knowing that your father has I feel , but throwing out


### Changing parameters
A) Paramenters: epochs 80 \ log-interval 100 \ emsize 300 \ nhid 300 \ dropout 0.3
	
A - Training
	
	| epoch   1 |   100/  233 batches | lr 20.00 | ms/batch 103.24 | loss  6.39 | ppl   594.56
	| epoch   1 |   200/  233 batches | lr 20.00 | ms/batch 102.05 | loss  5.33 | ppl   207.21
	-----------------------------------------------------------------------------------------
	| end of epoch   1 | time: 25.00s | valid loss  5.10 | valid ppl   164.73
	-----------------------------------------------------------------------------------------
	
	[...]
	
	-----------------------------------------------------------------------------------------
	| epoch  40 |   100/  233 batches | lr 0.00 | ms/batch 105.88 | loss  3.58 | ppl    35.98
	| epoch  40 |   200/  233 batches | lr 0.00 | ms/batch 103.77 | loss  3.43 | ppl    30.80
	-----------------------------------------------------------------------------------------
	| end of epoch  40 | time: 25.53s | valid loss  4.39 | valid ppl    80.67
	-----------------------------------------------------------------------------------------
	
	[...]
	
	-----------------------------------------------------------------------------------------
	| epoch  80 |   100/  233 batches | lr 0.00 | ms/batch 102.51 | loss  3.58 | ppl    35.93
	| epoch  80 |   200/  233 batches | lr 0.00 | ms/batch 101.50 | loss  3.43 | ppl    30.80
	-----------------------------------------------------------------------------------------
	| end of epoch  80 | time: 24.94s | valid loss  4.39 | valid ppl    80.67
	-----------------------------------------------------------------------------------------
	=========================================================================================
	| End of training | test loss  4.30 | test ppl    73.51
	=========================================================================================
	time taken:
	2042 seconds

A - Generated text
	
	till travel What they know from , for the world ... and a father &apos;ll be back here , like
	something kill Lois . <eos> Look , I &apos;m an smart man for a <unk> , but Stewie &apos;ll be
	five pieces . <eos> Well , you &apos;ve always asked me to pooping down . <eos> I &apos;m Tom Tucker
	&apos;s Peter Griffin , who the man nation her , I &apos;m seven advice a year . <eos> But or
	my daughters support him , you &apos;ve been getting knowing about your father , I don &apos;t know . <eos>


B) Parameters: epochs 40 \ log-interval 100 \ emsize 400 \ nhid 400 \ dropout 0.2
	
B - Training
	
	| epoch   1 |   100/  233 batches | lr 20.00 | ms/batch 148.47 | loss  6.41 | ppl   610.83
	| epoch   1 |   200/  233 batches | lr 20.00 | ms/batch 146.18 | loss  5.31 | ppl   202.21
	-----------------------------------------------------------------------------------------
	| end of epoch   1 | time: 35.77s | valid loss  5.07 | valid ppl   159.15
	-----------------------------------------------------------------------------------------
	
	[...]
	
	-----------------------------------------------------------------------------------------
	| epoch  20 |   100/  233 batches | lr 0.02 | ms/batch 156.85 | loss  3.53 | ppl    34.04
	| epoch  20 |   200/  233 batches | lr 0.02 | ms/batch 157.30 | loss  3.34 | ppl    28.19
	-----------------------------------------------------------------------------------------
	| end of epoch  20 | time: 37.82s | valid loss  4.44 | valid ppl    85.18
	-----------------------------------------------------------------------------------------
	
	[...]
	
	-----------------------------------------------------------------------------------------
	| epoch  40 |   100/  233 batches | lr 0.00 | ms/batch 150.09 | loss  3.52 | ppl    33.88
	| epoch  40 |   200/  233 batches | lr 0.00 | ms/batch 150.11 | loss  3.33 | ppl    28.08
	-----------------------------------------------------------------------------------------
	| end of epoch  40 | time: 36.55s | valid loss  4.44 | valid ppl    85.13
	-----------------------------------------------------------------------------------------
	=========================================================================================
	| End of training | test loss  4.34 | test ppl    76.93
	=========================================================================================
	time taken:
	3114 seconds
	
B - Generated text

	. <eos> What &apos;s on our family for the world ... and the Quahog C brought me a <unk> dry
	<unk> . <eos> I &apos;ve never had a lovely friend , and I like a few bucks or so I
	can &apos;t believe them and all those famous people have to become bad . <eos> <unk> I &apos;m Tom Tucker
	who has a veteran is a <unk> nation for having the problem . <eos> Who you have said the <unk>
	<unk> daughters the just pork and <unk> . <eos> The currently <unk> dealing of <unk> <unk> , is throwing over


C) Parameters: epochs 40 \ log-interval 100 \ emsize 100 \ nhid 100 \ dropout 0.4
	
C - Training
	
	| epoch   1 |   100/  233 batches | lr 20.00 | ms/batch 32.27 | loss  6.27 | ppl   527.36
	| epoch   1 |   200/  233 batches | lr 20.00 | ms/batch 30.92 | loss  5.44 | ppl   229.58
	-----------------------------------------------------------------------------------------
	| end of epoch   1 | time:  7.67s | valid loss  5.15 | valid ppl   171.62
	-----------------------------------------------------------------------------------------
	
	[...]
	
	-----------------------------------------------------------------------------------------
	| epoch  20 |   100/  233 batches | lr 5.00 | ms/batch 31.48 | loss  4.24 | ppl    69.09
	| epoch  20 |   200/  233 batches | lr 5.00 | ms/batch 30.88 | loss  4.16 | ppl    64.02
	-----------------------------------------------------------------------------------------
	| end of epoch  20 | time:  7.60s | valid loss  4.45 | valid ppl    85.75
	-----------------------------------------------------------------------------------------

	[...]
	
	-----------------------------------------------------------------------------------------
	| epoch  40 |   100/  233 batches | lr 0.02 | ms/batch 31.37 | loss  4.16 | ppl    64.22
	| epoch  40 |   200/  233 batches | lr 0.02 | ms/batch 31.06 | loss  4.09 | ppl    59.62
	-----------------------------------------------------------------------------------------
	| end of epoch  40 | time:  7.60s | valid loss  4.43 | valid ppl    83.62
	-----------------------------------------------------------------------------------------
	=========================================================================================
	| End of training | test loss  4.34 | test ppl    77.07
	=========================================================================================
	time taken:
	307 seconds

C - Generated text

	. <eos> What &apos;s on from this point , they have a <unk> for you to fall with there like
	something and Lois &apos;s been talking for a village &apos; , and I like a <unk> of or spread on
	five pieces . <eos> Well , you &apos;ve always like to fight pooping to the <unk> and with people who
	&apos;s going to sell out . <eos> Well , we &apos;t do it like a good beer in the <unk>
	for Hollywood old just , and I am gonna help someone who catches a <unk> gently . <eos> We pray


D) Parameters: epochs 100 \ log-interval 100 \ emsize 200 \ nhid 200 \ dropout 0.3
	
D - Training
	
	| epoch   1 |   100/  233 batches | lr 20.00 | ms/batch 63.19 | loss  6.28 | ppl   536.41
	| epoch   1 |   200/  233 batches | lr 20.00 | ms/batch 60.86 | loss  5.34 | ppl   208.63
	-----------------------------------------------------------------------------------------
	| end of epoch   1 | time: 15.10s | valid loss  5.07 | valid ppl   159.45
	-----------------------------------------------------------------------------------------
	
	[...]
	
	-----------------------------------------------------------------------------------------
	| epoch  50 |   100/  233 batches | lr 0.00 | ms/batch 60.81 | loss  3.66 | ppl    38.75
	| epoch  50 |   200/  233 batches | lr 0.00 | ms/batch 60.68 | loss  3.54 | ppl    34.52
	-----------------------------------------------------------------------------------------
	| end of epoch  50 | time: 14.79s | valid loss  4.39 | valid ppl    80.59
	-----------------------------------------------------------------------------------------
	
	[...]
	
	-----------------------------------------------------------------------------------------
	| epoch 100 |   100/  233 batches | lr 0.00 | ms/batch 61.44 | loss  3.65 | ppl    38.65
	| epoch 100 |   200/  233 batches | lr 0.00 | ms/batch 60.78 | loss  3.54 | ppl    34.52
	-----------------------------------------------------------------------------------------
	| end of epoch 100 | time: 14.89s | valid loss  4.39 | valid ppl    80.59
	-----------------------------------------------------------------------------------------
	=========================================================================================
	| End of training | test loss  4.29 | test ppl    73.09
	=========================================================================================
	time taken:
	3090 seconds
		
D - Generated text
	
	around an hour again . <eos> That &apos;s right , so if you were rather a child with me dry
	and kill yourself for a hot hole . <eos> A smart man for Brian <unk> , or Stewie Griffin &apos;s
	five pieces . <eos> Well , we &apos;ve got an issue of pooping , so we don &apos;t want to
	come through the sun . <eos> And now we can &apos;t do it like a good beer in the <unk>
	for Hollywood . <eos> I know , it &apos;s been currently about dealing on I <unk> up our throwing until

----
# Task 2

We worked on a macOS system, so eventually you need to make some changes if you run on a windows machine, for example change `python3`to `python`or disable mps. 

## Modifications of main.py for logging perplexity

Our modified main.py can be found in the folder `scripts` and is called `main_modified.py`. We added two additional arguments to the parser:

- `save-perplexities` the flag to determine if the perplexities should be stored in a log file
- `log-file` the path to store the log file

In the file you find a function called `log_perplexity()` which takes arguments as described in the code. This is called at the corresponding places in the script. All of the "Train", "Validation" and "Test" logs are stored in the same log file in the following structure: 

	1   Train  167.8992049241194  

## Model training

We trained 5 models with the following perplexities: 
- 0.0
- 0.2
- 0.4
- 0.6
- 0.8

For the other parameters we used the parameters as followed. These remained the same for all of the 5 models, only the dropout value changed. As log-interval we chose 10 to get close to the end of one epoch. 

	(  
	  cd $tools/pytorch-examples/word_language_model &&  
	    CUDA_VISIBLE_DEVICES=$device OMP_NUM_THREADS=$num_threads python3 main.py \  
	      --data $data/familyguy \  
	      --epochs 40 \  
	      --log-interval 10 \  
	      --emsize 200 --nhid 200 --dropout {dropout_value} --tied \  
	      --save $models/model.pt \  
	      --mps \  
	      --save-perplexities \  
	      --log-file $logs/perplexities_{dropout_value}.txt  
	)

## Clean and split log files

After training, we got 5 different log files in the folder called `logs`:
- `perplexities_0.0.txt`
- `perplexities_0.2.txt`
- `perplexities_0.4.txt`
- `perplexities_0.6.txt`
- `perplexities_0.8.txt`

Since our log-interval is 10 we get many logs for each epoch. Since only the log at the end of the epoch is relevant, we have to clean the log files to get only these specific numbers. Additionally to create the three tables for training, validation and test data, we have to split up the data. To do so, navigate to the `scripts` directory execute the following command in the terminal:

	python3 clean_log_folder.py --input_directory ../logs --output_directory ../logs_clean

This command takes the path to the directory of all of the log files (input) and the path where the cleaned log files should be stored (output).

The file extracts the last training log of each epoch and splits the data into training, validation and test files. In the end you have for each dropout value three files stored in `logs_clean/test`, `logs_clean/training` and `logs_clean/validation` such as:

- `perplexities_0.0_clean_test.txt`
- `perplexities_0.0_clean_train.txt`
- `perplexities_0.0_clean_valid.txt`

and this for every dropout value. 
## Create tables 

Since we now have the data split up to test, training and validation data, we can now create the tables. To do so, still in the `scripts` directory, execute the following command in the terminal:

	python3 create_log_table.py --test_dir ../logs_clean/test --valid_dir ../logs_clean/validation --train_dir ../logs_clean/training --output_dir ../logs_csv

You have to input the directories of the split up training, validation and test data and the output directory, where you want to store your tables.

For every directory the script creates a pandas dataframe as our table which we need for the visualization. We decided to store the dataframes as .csv files in the directory `logs_csv`. The script creates a dataframe for each training, validation and test data with the dropout values as headers and for each epoch the values. The tables look as follows:

	Validation Dataframe:
	       Dropout 0.0  Dropout 0.2  Dropout 0.4  Dropout 0.6  Dropout 0.8
	Epoch                                                                 
	1       161.108788   380.423364   182.937092   182.615540   248.844383
	2       117.240914   286.036730   126.706868   128.101396   165.601937
	3       102.137781   274.985956   110.983996   115.972402   151.404262
	4        96.319723   248.169602   106.529639   112.207522   146.764167
	[...]

	Training Dataframe:
       Dropout 0.0  Dropout 0.2  Dropout 0.4  Dropout 0.6  Dropout 0.8
	Epoch                                                                 
	1       137.991083   149.405749   149.821519   157.663685   201.523243
	2        93.579697   102.743097   108.196054   120.306297   164.836583
	3        74.828612    87.271733    93.523414   110.461468   156.139863
	4        64.239827    76.116960    86.162962   104.691018   151.153137
	[...]

	Test Dataframe:
       Dropout 0.0  Dropout 0.2  Dropout 0.4  Dropout 0.6  Dropout 0.8
	Epoch                                                                 
	40       80.835491   179.114961    74.338993    87.126646   121.397025

For the full tables please look at the corresponding .csv files in the `logs_csv` directory.

## Visualize the data

For the visualization still in the `scripts` directory execute the following command in the terminal:

	python3 plot_perplexity.py --validation_csv ../logs_csv/validation_dataframe.csv --training_csv ../logs_csv/training_dataframe.csv --output_dir ../plots

The script takes the training and validation .csv files (dataframes) and store the images of the plot in the specified output directory.

Our plots look as follows:

![Training Plot](plots/training_perplexity_plot.png)


![Validation Plot](plots/validation_perplexity_plot.png)

Final test perplexities:
- 0.0: 80.9
- 0.2: 179.1
- 0.4: 74.3
- 0.6: 87.1
- 0.8: 121.4

## Analysis of the results

Generally, both the training and validation perplexities should follow a similar trend such as decreasing while the model learns. If the training perplexity decreases and validation perplexity does not, it's a sign of overfitting. The test perplexity does a final check on the models performance and should align the validation trend. 

In our case, the final test perplexities align pretty well with the final test perplexities. The dropout of 0.4 has the lowest test perplexity and also the lowest validation perplexity which indicates that this is the best dropout setting to choose. It has not the lowest perplexity at training, but suggests that the dropout rate of 0.4 is better at generalization to the test data while e. g. the dropout rate of 0.0 is better at learning the training data (lower perplexity) but worse at generalization than the dropout rate of 0.4 (higher validation and final test perplexity). So we choose the dropout rate of 0.4.

**Sample text lowest perplexity**

	for a lot of city tiny <unk> and life ... like , now , I know you &apos;re going to  
	<unk> , blood , get tampon on his stupid Face . <eos> Except glad Mr <unk> up here I &apos;m  
	the only so <unk> . <eos> It just really hard than me , but I don &apos;t know what my  
	new days is then . <eos> I never sometimes or the options and even , like , Nobody helping each  
	other <unk> their own immediately that would you say you right . <eos> Coming up , as many smell is

**Sample data highest perplexity**

We got the worst perplexity with the dropout rate of 0.2:

	for a <unk> <unk> . light - a life inside today was a <unk> I <unk> finding a football .  
	<unk> , blood . <eos> tampon ? <eos> stupid in the other arm , Mr <unk> fake . ( multiracial  
	the students when <unk> . . face onto an drinks clean with a <unk> <unk> hours tribe . <eos> :  
	over . a reporter . the stuff ! sometimes or the options . even , I once . helping .  
	<eos> <unk> their <unk> ? <eos> Which . while the zoo way this sad is to , . smell it


We think the text with the lowest perplexity is better than the one with the highest perplexity. It has almost complete sentences and we think the text is a bit humorous ("stupid face") which resembles the training data. The text with the highest perplexity ist not very structured, has a lot of random punctuation in it and still has a lot of "unk" words in it. 
