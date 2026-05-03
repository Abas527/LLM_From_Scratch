from src.generate import pipeline

if __name__ == "__main__":
    input_text="once there was"
    output_text=pipeline(input_text)
    print(output_text)