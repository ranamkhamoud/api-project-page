import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoConfig, AutoModel, PreTrainedModel
from pathlib import Path
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
# nothing is random here so no seed is set

# code used from https://huggingface.co/desklib/ai-text-detector-v1.01 and modified for this project
class DesklibAIDetectionModel(PreTrainedModel):
    config_class = AutoConfig

    def __init__(self, config):
        # Initialize the PreTrainedModel
        super().__init__(config)
        # Initialize the base transformer model.
        self.model = AutoModel.from_config(config)
        # Define a classifier head.
        self.classifier = nn.Linear(config.hidden_size, 1)
        # Initialize weights (handled by PreTrainedModel)
        self.init_weights()

    def forward(self, input_ids, attention_mask=None, labels=None):
        # Forward pass through the transformer
        outputs = self.model(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs[0]
        # Mean pooling
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_expanded, dim=1)
        sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
        pooled_output = sum_embeddings / sum_mask

        # Classifier
        logits = self.classifier(pooled_output)
        loss = None
        if labels is not None:
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(logits.view(-1), labels.float())

        output = {"logits": logits}
        if loss is not None:
            output["loss"] = loss
        return output

def predict_single_text(text, model, tokenizer, device, max_len=768, threshold=0.5):
    encoded = tokenizer(
        text,
        padding='max_length',
        truncation=True,
        max_length=max_len,
        return_tensors='pt'
    )
    input_ids = encoded['input_ids'].to(device)
    attention_mask = encoded['attention_mask'].to(device)

    model.eval()
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs["logits"]
        probability = torch.sigmoid(logits).item()

    ai_detected = True if probability >= threshold else False
    return probability, ai_detected

# own code to easily create text files, and feed them to the model for predictions
def ai_plagiarism_detection(text, threshold=0.5, show_results=False):
    """
    Detect if the given text is AI generated or human written.
    Args:
        text (str): Input text to be classified.
        show_results (bool): If True, prints the results.
    Returns:
        probability (float): Probability of being AI generated.
        ai_detected (bool): True if AI generated, Falce if human written.
    """

    # Model and Tokenizer Directory
    model_directory = "desklib/ai-text-detector-v1.01"
    # Load tokenizer and model
    tokenizer = AutoTokenizer.from_pretrained(model_directory)
    model = DesklibAIDetectionModel.from_pretrained(model_directory)
    # Set up device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    # Predict
    probability, ai_detected = predict_single_text(text, model, tokenizer, device, threshold=threshold)
    # to print results
    if show_results:
        print(f"Probability of being AI generated: {probability:.4f}")
        print(f"Predicted label: {'AI Generated' if ai_detected else 'Not AI Generated'}")
    return probability, ai_detected


def make_textfile(file_path="text_folder/example.txt", content = "This is an example text file.\nAnd this is the second line.\n"):
    """
    Create a text file with the given content.
    Args:
        file_path (str): Path to the text file to be created.
        content (str): Content to write into the text file.
    """
    # Open the file in write mode ('w') and write some content
    with open(file_path, "w") as f:
        f.write(content)
    return

def get_text_from_textfile(text_dir="text_folder"):
    """
    Read all text files from a directory and return a dictionary with filename as key and content as value.
    Args:
        text_dir (str): Directory containing text files.
    Returns:
        text_dict (dict): Dictionary with filename as key and file content as value.
    """
    text_dict = {}
    text_file_list = list(Path(text_dir).glob("*.txt"))
    for elem in text_file_list:
        content = elem.read_text(encoding="utf-8")  # read file content
        text_dict[elem.name] = content  # use filename as key
    return text_dict

def classifying_plagiarism_using_textfiles(best_threshold=0.78):
    """
    This function shows how this model can be used to detect ai in the text files in the text_folder folder. This is what is to be used in the pipeline.
    """
    # make sure folder exists
    Path("text_folder").mkdir(exist_ok=True)
    
    # create example text files
    make_textfile("text_folder/ai_text.txt", "AI detection refers to the process of identifying whether a given piece of content, such as text, images, or audio, has been generated by artificial intelligence. This is achieved using various machine learning techniques, including perplexity analysis, entropy measurements, linguistic pattern recognition, and neural network classifiers trained on human and AI-generated data. Advanced AI detection tools assess writing style, coherence, and statistical properties to determine the likelihood of AI involvement. These tools are widely used in academia, journalism, and content moderation to ensure originality, prevent misinformation, and maintain ethical standards. As AI-generated content becomes increasingly sophisticated, AI detection methods continue to evolve, integrating deep learning models and ensemble techniques for improved accuracy.")  # create an example text file
    make_textfile("text_folder/human_text.txt", "It is estimated that a major part of the content in the internet will be generated by AI / LLMs by 2025. This leads to a lot of misinformation and credibility related issues. That is why if is important to have accurate tools to identify if a content is AI generated or human written")  # create another example text file
    textfile_dict = get_text_from_textfile(text_dir="text_folder")  # get dict with text file and content, text_dir is folder containing text files that need to be classified

    # get predictions for each text file
    for textfile, text in textfile_dict.items(): # for key, value in ft_dict.items():
        print(f"Getting predictions for: {textfile}")
        # ---------- GET PREDICTIONS ----------
        probability, ai_detected = ai_plagiarism_detection(text=text, threshold=best_threshold, show_results=False) # get predictions with the optimal threshold value: 0.78
        # print results
        print(f"{textfile} Results:\n Probability of being AI generated: {probability:.4f}")
        print(f" Predicted label: {'AI Generated' if ai_detected else 'Not AI Generated'}\n")



def get_texts_from_jsonfile(json_file_path, sample_size=100, ignore_warning=False):
    """
    Get text partitions from a json file. Each partition is a text that can be given as input to the ai_plagiarism_detection model.
    Args:
        json_file_path (str): Path of the json file.
        sample_size (int): Determines how many batches are returned.
    Returns:
        text_list (list): All the text batches in order of the json file as elements in a list.
    """
    text_list = []
    try:
        with open(json_file_path, "r", encoding="utf-8") as f:
            for i, line in enumerate(f):
                obj = json.loads(line)
                text_list.append(obj["text"])
                if i == sample_size-1:
                    break
    except:
        raise ValueError(f"{json_file_path} does not exist or is not found.")
    # raise warning if less texts found than sample size
    if ignore_warning != True:
        if len(text_list) != sample_size:
            raise ValueError(f"Warning: only {len(text_list)} texts found, less than sample size {sample_size}")

    return text_list

def run_experiment_using_jsonfile(threshold=0.5):
    """
    This function runs the experiment and saves the results in ai_plagiarism_experiment/ai_plagiarism_detection_results.csv
    """
    # Set Total sample size, there are two datasets (json's) used, so sample_size//2 per dataset is used.
    sample_size = 240 
    sample_size //=2
    

    # make sure folders exist
    Path("json_folder").mkdir(exist_ok=True)
    Path("ai_plagiarism_experiment").mkdir(exist_ok=True)

    # ------- GET TRUE NEGATIVE TEXTS (human thought and spoken) FROM JSON FILE -------
    # load json file with text whisper transribed text from ML commons dataset
    text_list = get_texts_from_jsonfile("json_folder/ML_commons.json", sample_size)
    
    # get predictions for each 
    predictions=[]
    for i, text in enumerate(text_list):
        # ---------- GET PREDICTIONS ----------
        probability, ai_detected = ai_plagiarism_detection(text=text, threshold=threshold, show_results=False)
        # save results
        predictions.append({"ML_commons_text_index": i,
                            "GPT_text_index": np.nan,
                            "text_length": len(text),
                            "topic": "unknown",
                            "probability": probability,
                            "ai_detected": ai_detected,
                            "really_ai": False
                            })
    # convert to dataframe
    df = pd.DataFrame(predictions)
    print("-------- 50% of samples predicted of json experiment --------")
    
    # ------- GET TRUE POSITIVE TEXTS (ai written) FROM JSON FILE -------
    # load json file with gpt generated texts
    text_list = get_texts_from_jsonfile("json_folder/gpt_generated.json", sample_size)

    predictions=[]
    for i, text in enumerate(text_list):
        # ---------- GET PREDICTIONS ----------
        probability, ai_detected = ai_plagiarism_detection(text=text, threshold=threshold, show_results=False)
        # # print results
        # print(f"Text {i} Results:\n Probability of being AI generated: {probability:.4f}")
        # print(f" Predicted label: {'AI Generated' if ai_detected else 'Not AI Generated'}\n")

        # save results
        if i < 40:
            topic = "astronomy"
        elif i < 80:
            topic = "quantum computing"
        else:
            topic = "daily life, personal growth, and everyday experiences"

        predictions.append({"ML_commons_text_index": np.nan,
                            "GPT_text_index": i,
                            "text_length": len(text),
                            "topic": topic,
                            "probability": probability,
                            "ai_detected": ai_detected,
                            "really_ai": True
                            })
    # convert to dataframe
    new_rows = pd.DataFrame(predictions)
    df = pd.concat([df, new_rows], ignore_index=True)
    print("------- 100% of samples predicted of json experiment --------")
    # save to csv
    df.to_csv("ai_plagiarism_experiment/ai_plagiarism_detection_results.csv", index=False)

    # update metrics
    get_metrics(threshold=threshold)


def get_metrics(df=None, threshold=0.5, save_to_csv=True):
    """
    This function calculates the metrics and saves them in ai_plagiarism_experiment/res_metrics(t={threshold}).csv
    """

    if df is None:
        # read from csv
        df = pd.read_csv("ai_plagiarism_experiment/ai_plagiarism_detection_results.csv")

    # calculate metrics
    fp = ((df["probability"]>=threshold) & (df["really_ai"]==False)).sum()  # false positives, cause all texts are human thought texts, however whisper makes text look more ai like
    tn = ((df["probability"]<threshold) & (df["really_ai"]==False)).sum()   # true negatives
    tp = ((df["probability"]>=threshold) & (df["really_ai"]==True)).sum()   # true positives
    fn = ((df["probability"]<threshold) & (df["really_ai"]==True)).sum()    # false negatives

    recall = tp/(tp+fn) if (tp+fn) != 0 else 0
    precision = tp/(tp+fp) if (tp+fp) != 0 else 0
    accuracy = (tp+tn)/(tp+fp+tn+fn) if (tp+fp+tn+fn) != 0 else 0

    # info of text lengths of both datasets
    ML_commons_length_mean = df.loc[df["ML_commons_text_index"].notna(), "text_length"].mean()
    ML_commons_length_std = df.loc[df["ML_commons_text_index"].notna(), "text_length"].std()
    gpt_length_mean = df.loc[df["GPT_text_index"].notna(), "text_length"].mean()
    gpt_length_std = df.loc[df["GPT_text_index"].notna(), "text_length"].std()
        

    # save metrics in dataframe
    results = pd.DataFrame({
        "Metric": ["TP", "TN", "FP", "FN", "Recall", "Precision", "Accuracy", "Total samples", "ML_commons_length_mean", "ML_commons_length_std", "gpt_length_mean", "gpt_length_std"],
        "Value": [tp, tn, fp, fn, recall, precision, accuracy, len(df), ML_commons_length_mean, ML_commons_length_std, gpt_length_mean, gpt_length_std]
    })
    if save_to_csv:
        # save in csv
        results.to_csv(f"ai_plagiarism_experiment/res_metrics(t={threshold}).csv", index=False) 
    return results
    
def tune_threshold(metric = "Accuracy"):
    """This function maximises the accuracy of the ai plagiarism detector given the results.csv"""

    df = pd.read_csv("ai_plagiarism_experiment/ai_plagiarism_detection_results.csv")
    # set boundaries
    min = 0.0
    max = 1.0
    step = 0.01
    # init
    best_accuracy=0
    m_l=[]
    t_l=[]
    for threshold in np.arange(min, max+step, step):
        threshold = round(threshold, 2)
        results = get_metrics(df,threshold,False)
        opti_metric = results.loc[results["Metric"] == metric, "Value"].iloc[0]
        # save for plotting
        m_l.append(opti_metric)
        t_l.append(threshold)
        # update best threshold
        if opti_metric>best_accuracy:
            best_accuracy = opti_metric
            best_threshold = threshold

    # plot tuning
    Path("ai_plagiarism_tuning_plots").mkdir(exist_ok=True)
    plt.plot(t_l, m_l)
    plt.xlabel("threshold")
    plt.ylabel(metric)
    plt.title(f"threshold vs {metric}")
    plt.savefig(f"ai_plagiarism_tuning_plots/threshold_vs_{metric}.png")
    plt.close()
    return best_threshold


        
        

if __name__ == "__main__":
    print("-------- Starting ai plagiarism experiment! --------\n")    
    # run experiment using json files
    run_experiment_using_jsonfile(threshold=0.5) # firstly using the default threshold

    # search for the theshold that maximises accuracy
    metric = "Accuracy"
    best_threshold_accuracy = tune_threshold(metric=metric)
    print(f"Best theshold for {metric}: {best_threshold_accuracy}")
    # search for the theshold that maximises precision
    metric = "Precision"
    best_threshold_precision = tune_threshold(metric=metric)
    print(f"Best theshold for {metric}: {best_threshold_precision}")

    # run experiment using json files
    run_experiment_using_jsonfile(threshold=best_threshold_accuracy) # secondly using the optimal threshold, the end result is 
    
    # example of usage that is fit for a pipeline using the best accuracy (best_threshold=0.78), when using best precision use best_threshold=0.97
    classifying_plagiarism_using_textfiles(best_threshold=best_threshold_accuracy)

    print("\n-------- Done! -------- ")
