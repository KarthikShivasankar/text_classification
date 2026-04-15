import gradio as gr
import pandas as pd
import os
import numpy as np
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import load_dataset, Dataset , load_from_disk
import torch
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from datasets import DatasetDict
from scipy.special import softmax
import torch
import gc 


# Define a directory for saved models and datasets if it does not exist
model_dir = './saved_models'
dataset_dir = './saved_datasets'

if not os.path.exists(model_dir):
    os.makedirs(model_dir)
if not os.path.exists(dataset_dir):
    os.makedirs(dataset_dir)

def list_saved_models():
    """Lists saved models for selection."""
    return [d for d in os.listdir(model_dir) if os.path.isdir(os.path.join(model_dir, d))]



def list_saved_datasets():
    """Lists saved datasets for selection."""
    return [d for d in os.listdir(dataset_dir) if os.path.isdir(os.path.join(dataset_dir, d))]


def process_and_save_csv(file_path, train_percent, min_word_length):
    """This function saves a processed CSV file for fine-tuning, removes NaN values, resets the index,
       filters text by minimum word length, and splits the dataset into training and test sets based on the specified training percentage."""
    try:
        df = pd.read_csv(file_path)
        
        if 'text' not in df.columns or 'label' not in df.columns:
            return "Error: CSV must contain 'text' and 'label' columns.", pd.DataFrame(), pd.DataFrame()
        
        df.dropna(inplace=True)
        df.reset_index(inplace=True)

        # Filter texts shorter than the specified minimum word length
        df['word_count'] = df['text'].apply(lambda x: len(x.split()))
        df = df[df['word_count'] >= min_word_length]
        df.drop(columns=['word_count'], inplace=True)

        if 'Unnamed: 0' in df.columns:
            df = df.drop('Unnamed: 0', axis=1)

        df = df.sample(frac=1, random_state=42).reset_index(drop=True)

        if train_percent < 100:
            train_size = int(len(df) * (train_percent / 100))
            train_df = df[:train_size]
            test_df = df[train_size:]

            # Define the dataset path
            dataset_name = "Finetune_Dataset"
            dataset_path = os.path.join(dataset_dir, dataset_name)
            os.makedirs(dataset_path, exist_ok=True)

            # Saving datasets to disk
            Dataset.from_pandas(train_df).save_to_disk(os.path.join(dataset_path, "train"))
            Dataset.from_pandas(test_df).save_to_disk(os.path.join(dataset_path, "test"))
        else:
            train_df = df  # Use the entire dataset as the training set

            # Define the dataset path
            dataset_name = "Evaluate_Dataset"
            dataset_path = os.path.join(dataset_dir, dataset_name)
            os.makedirs(dataset_path, exist_ok=True)

            # Saving the training dataset to disk only
            Dataset.from_pandas(train_df).save_to_disk(os.path.join(dataset_path, "train"))

        
        return "CSV Processed" , train_df.head()
    except Exception as e:
        return f"Failed to process the file: {str(e)}", pd.DataFrame(), pd.DataFrame()



def fine_tune_model(model_name):
    """
 it uses the selected dataset.
    """
 
    
    dataset_train_path = os.path.join(dataset_dir, "Finetune_Dataset", 'train')

          # Load only the train split of the dataset
    if os.path.exists(dataset_train_path):
        dataset = Dataset.load_from_disk(dataset_train_path)
    
    # Load model and tokenizer
    model_path = f"./saved_models/{model_name}"

    if os.path.exists(model_path) and model_name == "binary_classification_train_TD":

        output_dir = "./saved_models/finetuned_TD"
        class_names = ['non_TD', 'TD']

    
    elif os.path.exists(model_path) and model_name == "High_priority_roberta":

        output_dir = "./saved_models/finetuned_Priority"
        class_names = ['not_High_priority', 'High_priority']

    tokenizer = AutoTokenizer.from_pretrained(model_path)
    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)  

    tokenized_datasets = dataset.map(tokenize_function, batched=True)

    tokenized_datasets = tokenized_datasets.train_test_split(test_size=0.020)  # Adjust the split ratio as needed
    tokenized_datasets = DatasetDict({
        'train': tokenized_datasets['train'],
        'test': tokenized_datasets['test']
    })

    
    # Fine-tuning
    training_args = TrainingArguments(
        output_dir=output_dir,
        evaluation_strategy="epoch",  # or use "steps" to evaluate more frequently
        learning_rate=2e-5,
        per_device_train_batch_size=4,
        num_train_epochs=2,
        save_strategy="no",
         # Load the best model at the end of training
    )
    
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=tokenized_datasets["train"],
        eval_dataset=tokenized_datasets["test"],  # Specify the evaluation dataset here
    )
    
    trainer.train()
    trainer.save_model() 
    tokenizer.save_pretrained(training_args.output_dir)

    dataset_test_path = os.path.join(dataset_dir, "Finetune_Dataset", 'test')

    if os.path.exists(dataset_test_path):
        test_dataset = Dataset.load_from_disk(dataset_test_path)
        print("Test dataset loaded successfully.")
    def tokenize_function(examples):
        return tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512)

    test_dataset = test_dataset.map(tokenize_function, batched=True)
    test_dataset.set_format(type='torch', columns=['input_ids', 'attention_mask', 'label'])
    

    output = trainer.predict(test_dataset)
    logits = output.predictions
    probs = softmax(logits, axis=1)
    predictions = np.argmax(logits, axis=1)
    positive_class_probs = probs[:, 1]
    # Metrics and Confusion Matrix
    labels = test_dataset['label']
    accuracy = accuracy_score(labels, predictions)
    conf_matrix = confusion_matrix(labels, predictions)
    class_names = class_names 
    report = classification_report(labels, predictions, target_names=class_names, output_dict=True)

    # Convert the report dictionary to a DataFrame
    df_classification_report = pd.DataFrame(report).transpose()

    # Optionally, you can reset the index to have the class labels as a column
    df_classification_report.reset_index(inplace=True)
    df_classification_report.rename(columns={'index': 'class'}, inplace=True)
    
# Format the floating point columns to three decimal places directly
    float_columns = ['precision', 'recall', 'f1-score', 'support']
    df_classification_report[float_columns] = df_classification_report[float_columns].applymap(lambda x: f"{x:.3f}")


    # Update dataset with predictions
    dataset_df = pd.DataFrame(test_dataset.remove_columns(['input_ids', 'attention_mask']).to_pandas())
    dataset_df['predicted'] = predictions
    dataset_df['probability'] = positive_class_probs
    dataset_head = dataset_df.head()

    # Save updated dataframe to CSV for download
    csv_path = "updated_dataset.xlsx"
    dataset_df.to_excel(csv_path, index=True)
    
    # Plot confusion matrix
    fig, ax = plt.subplots(figsize=(8,6))
    sns.heatmap(conf_matrix, annot=True, fmt='d', ax=ax, cmap="Blues", xticklabels=class_names, yticklabels=class_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')
    plt.close(fig)  # Prevent the figure from displaying immediately

    return "Model fine-tuned." , accuracy, df_classification_report , gr.Plot(fig), dataset_head, csv_path
# Mapping of model types to their respective directories
model_mapping = {
    "TD": "binary_classification_train_TD",
    "FT_TD": "finetuned_TD",
    "High_Priority": "High_priority_roberta",
     "FT_HighPriority": "finetuned_Priority",
    "Quality": [
        "binary_classification_train_comp", "binary_classification_train_main",
        "binary_classification_train_perf", "binary_classification_train_port",
        "binary_classification_train_reli", "binary_classification_train_secu",
        "binary_classification_train_usab"
    ],
    "Types_TD": [
        "binary_classification_train_automation", "binary_classification_train_build",
        "binary_classification_train_design", "binary_classification_train_documentation",
        "binary_classification_train_infrastructure", "binary_classification_train_people",
        "binary_classification_train_test", "binary_classification_train_architecture"
    ]
}

def setup_tokenizer():
    first_model_dir = next(iter(model_mapping.values()))
    if isinstance(first_model_dir, list):
        first_model_dir = first_model_dir[0]
    tokenizer_path = os.path.join(model_dir, first_model_dir)
    return AutoTokenizer.from_pretrained(tokenizer_path)

tokenizer = setup_tokenizer()

# Function to load model and set it to use GPU
def load_model_and_tokenizer(model_directory):
    model_path = os.path.join(model_dir, model_directory)
    if not os.listdir(model_path):
        print(f"No model files found in {model_path}")
        return None
    try:
        model = AutoModelForSequenceClassification.from_pretrained(model_path)
        if torch.cuda.is_available():
            model.cuda()  # Move model to GPU
        return model
    except Exception as e:
        print(f"Failed to load model from {model_path}: {e}")
        return None

def load_and_tokenize_dataset():
    dataset_test_path = os.path.join(dataset_dir, "Evaluate_Dataset", 'train')
    if os.path.exists(dataset_test_path):
        test_dataset = load_from_disk(dataset_test_path)
        print(test_dataset)
        print("Test dataset loaded successfully.")
        return test_dataset.map(lambda examples: tokenizer(examples["text"], padding="max_length", truncation=True, max_length=512), batched=True)
    else:
        print("Dataset path does not exist.")
        return None

def evaluate_model(model_types, test_dataset):

    print(model_types)
    if test_dataset is None:
        print("No dataset available for evaluation.")
        return pd.DataFrame(), "No dataset loaded"
    
    print(test_dataset)


    results_df = pd.DataFrame(test_dataset.remove_columns(['input_ids', 'attention_mask']).to_pandas())
    print(len(results_df))
    for model_type in model_types:
        model_directories = model_mapping.get(model_type, [])
        if not isinstance(model_directories, list):
            model_directories = [model_directories]
            print(model_directories)

        for model_dir in model_directories:
            print(model_dir)
            model = load_model_and_tokenizer(model_dir)
            if model is None:
                continue

            inputs = tokenizer(test_dataset['text'], return_tensors="pt", padding=True, truncation=True, max_length=512)
            inputs = inputs.to('cuda')  # Move inputs to GPU
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=1)
                probabilities = torch.nn.functional.softmax(outputs.logits, dim=1)

            results_df[f'Prediction_{model_dir}'] = predictions.cpu().numpy()  # Move predictions back to CPU
            results_df[f'Probability_{model_dir}'] = probabilities.max(dim=1).values.cpu().numpy()

            del model
            del inputs
            torch.cuda.empty_cache()  # Explicitly clears up unused memory
            gc.collect()  # Collects garbage

    excel_path = "evaluation_results.xlsx"
    results_df.to_excel(excel_path, index=True)

    return results_df.head(), excel_path

with gr.Blocks() as demo:


    with gr.Tab("Fine-tune Model"):
        
        
        csv_upload = gr.File(label="Upload CSV for Fine-tuning")
        train_percent = gr.Slider(label="Training Set Percentage", minimum=10, maximum=100, step=5, value=30)
        min_word_length = gr.Slider(label="Minimum Word Length", minimum=15, maximum=100, step=1, value=45)

        csv_status = gr.Textbox(label="CSV Status")

        process_button = gr.Button("Process CSV")
        train_dataframe_head_output = gr.Dataframe(label="Training Data Head")
             
        model_selection = gr.Dropdown(label="Select a Model", choices=["binary_classification_train_TD", "High_priority_roberta"])
        fine_tune_button = gr.Button("Fine-tune Model")
        fine_tune_status = gr.Textbox(label="Fine-tuning Status")
        accuracy_output = gr.Textbox(label="Accuracy")
        classification_report_output = gr.Dataframe(label="Classification Report")
        confusion_matrix_plot = gr.Plot(label="Confusion Matrix")
        dataframe_head_output = gr.Dataframe(label="Dataset Head")
        dataframe_download = gr.File(label="Download Dataset with Predictions", type="filepath" , scale=2 )
        process_button.click(
            fn=process_and_save_csv,
            inputs=[csv_upload, train_percent, min_word_length],
            outputs=[csv_status, train_dataframe_head_output]
        )

        fine_tune_button.click(
            fn=fine_tune_model,
            inputs=[model_selection],
            outputs=[fine_tune_status , accuracy_output, classification_report_output, confusion_matrix_plot, dataframe_head_output, dataframe_download]
        )

        # Update visibility based on process output


    with gr.Tab("Evaluate Model"):
        model_type_selection = gr.CheckboxGroup(
        label="Select Model Types for Evaluation",
        choices=list(model_mapping.keys()),
        value=["TD"]  # You can set default values if needed
)
       
        csv_upload = gr.File(label="Upload CSV for Evaluating")
        train_percent = gr.Slider(label="Training Set Percentage", minimum=100, maximum=100, step=5, value=100)
        min_word_length = gr.Slider(label="Minimum Word Length", minimum=15, maximum=100, step=1, value=45)

        csv_status = gr.Textbox(label="CSV Status")
        process_button = gr.Button("Process CSV")
        train_dataframe_head_output = gr.Dataframe(label="Training Data Head")
        process_button.click(
            fn=process_and_save_csv,
            inputs=[csv_upload, train_percent, min_word_length],
            outputs=[csv_status, train_dataframe_head_output]
        )
      
        
        evaluate_button = gr.Button("Evaluate Model" )
        dataframe_head_output = gr.Dataframe(label="Dataset Head")
        dataframe_download = gr.File(label="Download Dataset with Predictions", type="filepath" , scale=3 )
        

        
        evaluate_button.click(
            fn=lambda model_types: evaluate_model(model_types, load_and_tokenize_dataset()),
            inputs=[model_type_selection],
            outputs=[dataframe_head_output, dataframe_download]
        )


demo.launch(server_name="0.0.0.0" , server_port=7077)


