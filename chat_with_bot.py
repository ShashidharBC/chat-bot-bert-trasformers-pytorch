from transformers import BertTokenizer, BertForSequenceClassification
import torch

def get_response(user_input, fine_tuned_model, tokenizer):
    inputs = tokenizer(user_input, return_tensors="pt", max_length=128, truncation=True, padding=True)
    outputs = fine_tuned_model(**inputs)
    predicted_class = torch.argmax(outputs.logits)
    print('THE CLASS>>>>', predicted_class)
    if predicted_class == 1:  # Bot response
        bot_response = tokenizer.decode(inputs["input_ids"][0], skip_special_tokens=True)  # Convert input_ids to text
        return "Nurse: " + bot_response
    else:
        return "Nurse: Sorry, I didn't understand that."

def main():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    fine_tuned_model = BertForSequenceClassification.from_pretrained("fine_tuned_chatbot_model")

    print("Nurse: Hi, I’m an AI assistant and I’m here to help you with diagnosis of your patient virtually.")
    while True:
        user_input = input("Doctor: ")
        if user_input.lower() == 'quit':
            print("Nurse: Goodbye!")
            break
        response = get_response(user_input, fine_tuned_model, tokenizer)
        print(response)

if __name__ == "__main__":
    main()
