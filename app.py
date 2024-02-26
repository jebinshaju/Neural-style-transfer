import streamlit as st
import os
import subprocess
import sys
import threading

def save_uploaded_file(uploaded_file, save_dir, filename):
    with open(os.path.join(save_dir, filename), "wb") as f:
        f.write(uploaded_file.getbuffer())

def run_uitest_file(progress_bar):
    python_executable = sys.executable
    subprocess.run([python_executable, "main.py"])
    progress_bar.progress(100)


   

    

def load_value():
    try:
        # Load the current value from the text file
        with open('variable_value.txt', 'r') as file:
            current_value = int(file.read())
    except FileNotFoundError:
        # If the file doesn't exist, initialize with a default value
        current_value = 5
    return current_value

def update_value(new_value):
    # Update the value in the text file
    with open('variable_value.txt', 'w') as file:
        file.write(str(int(new_value)))
def main():
    st.title("Neural Style Transfer")
    
   
    content_uploaded_file = st.file_uploader("Upload Content Image", type=None)
    
    if content_uploaded_file is not None:
        save_dir_content = "content"
        if not os.path.exists(save_dir_content):
            os.makedirs(save_dir_content)
        save_uploaded_file(content_uploaded_file, save_dir_content, "contentpic.jpg")
        st.success("Content image uploaded successfully!")
        st.image("content/contentpic.jpg", caption="Content Image")

    
    style_uploaded_file = st.file_uploader("Upload Style Image", type=None)
    
    if style_uploaded_file is not None:
        save_dir_style = "style"
        if not os.path.exists(save_dir_style):
            os.makedirs(save_dir_style)
        save_uploaded_file(style_uploaded_file, save_dir_style, "stylepic.jpg")
        st.success("Style image uploaded successfully!")
        st.image("style/stylepic.jpg", caption="Style Image")

        # Load the current value from the text file
    current_value = load_value()

    # Display the slider widget
    new_value = st.slider('Adjust the strength:', 1, 10, current_value)

    # Update the value in the text file if it's changed
    if new_value != current_value:
        update_value(new_value)
        st.success('Value updated successfully!')

    
    if st.button("Transfer Style"):
        st.write("Processing....")
        progress_bar = st.progress(0)  # Create progress bar
        thread = threading.Thread(target=run_uitest_file, args=(progress_bar,))
        thread.start()
        thread.join()  # Wait for the thread to finish
        st.success("Image Generated.")

        if os.path.exists("generated_image.jpg"):
            st.image("generated_image.jpg", caption="Generated Image")
        else:
            st.error("No generated image found!")

if __name__ == "__main__":
    main()
