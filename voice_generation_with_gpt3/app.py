import openai
import boto3
import os

# Set your OpenAI API Key and AWS Credentials here.
openai.api_key = os.getenv("OPENAI_API_KEY")  
aws_access_key = os.getenv("AWS_ACCESS_KEY_ID")  
aws_secret_key = os.getenv("AWS_SECRET_ACCESS_KEY") 

# Initialize AWS Polly (Text to Speech service)
polly = boto3.client(
    'polly',
    aws_access_key_id=aws_access_key,
    aws_secret_access_key=aws_secret_key,
    region_name='us-east-1'  
)

# Function to call GPT-3 and generate text from a given prompt
def generate_text(prompt):
    try:
        response = openai.Completion.create(
            engine="text-davinci-003",  # Using GPT-3 here; you can explore other models
            prompt=prompt,
            max_tokens=150
        )
        # Returning the generated text
        return response.choices[0].text.strip()
    except Exception as e:
        print(f"Error generating text: {e}")
        return None

# Convert generated text to speech using AWS Polly
def text_to_speech(text):
    try:
        response = polly.synthesize_speech(
            Text=text,
            OutputFormat='mp3',  
            VoiceId='Joanna'  
        )
        return response['AudioStream']  
    except Exception as e:
        print(f"Error synthesizing speech: {e}")
        return None

# Main function to tie everything together
def main():
    # Take user input as the prompt for GPT-3
    prompt = input("Enter a prompt for the voice generation: ")
    
    # Generate text using GPT-3
    generated_text = generate_text(prompt)
    
    if generated_text:
        print(f"Generated Text: {generated_text}")
        
        # Convert the generated text into speech
        speech = text_to_speech(generated_text)
        
        if speech:
            # Write the audio stream to an MP3 file
            with open('output.mp3', 'wb') as file:
                file.write(speech.read())
            print("Speech saved as 'output.mp3'")
        else:
            print("Speech synthesis failed.")
    else:
        print("Text generation failed.")

# Entry point of the application
if __name__ == "__main__":
    main()
