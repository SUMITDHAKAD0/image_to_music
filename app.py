import streamlit as st

import numpy as np
import pandas as pd

import cv2
import time
import random
from PIL import Image
from scipy.io import wavfile
import IPython.display as ipd
from img_to_music import image_to_music, makeScale

# Adding an appropriate title for the test website
st.title("Making Music From Images")

st.markdown("Select sample image if you'd like to use one of the preloaded images. Select User Image is you'd like to use your own image.")
#Making dropdown select box containing scale, key, and octave choices
df1 = pd.DataFrame({'Scale_Choice': ['AEOLIAN', 'BLUES', 'PHYRIGIAN', 'CHROMATIC','DORIAN','HARMONIC_MINOR','LYDIAN','MAJOR','MELODIC_MINOR','MINOR','MIXOLYDIAN','NATURAL_MINOR','PENTATONIC']})
df2 = pd.DataFrame({'Keys': ['A','a','B','C','c','D','d','E','F','f','G','g']})
df3 = pd.DataFrame({'Octaves': [1,2,3]})
df4 = pd.DataFrame({'Harmonies': ['U0','ST','M2','m3','M3','P4','DT','P5','m6','M6','m7','M7','O8']})

st.sidebar.markdown("This app converts an image into a song. Play around with the various inputs belows using different images!")

sr_value = st.sidebar.number_input('SR value', value=22050)
scale = st.sidebar.selectbox('What scale would you like yo use?', df1['Scale_Choice'])
key = st.sidebar.selectbox('What key would you like to use?', df2['Keys']) 
octave = st.sidebar.selectbox('What octave would you like to use?', df3['Octaves']) 
harmony = st.sidebar.selectbox('What harmony would you like to use?', df4['Harmonies']) 
t_value = st.sidebar.slider('Note duration [s]', min_value=0.01, max_value=1.0, step = 0.01, value=0.2)     
n_pixels = st.sidebar.slider('How many pixels to use? (More pixels take longer)', min_value=12, max_value=320, step=1, value=60)  
 #Ask user if they want to use random pixels
random_pixels = st.sidebar.checkbox('Use random pixels to build song?', value=True)

#Ask user to select song duration
use_octaves = st.sidebar.checkbox('Randomize note octaves while building song?', value=True) 


col1, col2 = st.columns(2)

with col1:
    #Load image 
    img2load = st.file_uploader(label="Upload your own Image", )

with col2:
    #Display the image
    if img2load != None:
        image = Image.open(img2load)
        new_image = image.resize((200, 200))
        st.image(new_image)    

def generate_music(img, octave, key, scale, sr_value, t_value, random_pixels, use_octaves, n_pixels, harmony):
    scaled = makeScale(octave, key, scale)
    
    #Make the song!
    song, song_df, harmony = image_to_music(img, 
                                           scale = scaled, 
                                           sr = sr_value,
                                           T = t_value, 
                                           randomPixels = random_pixels, 
                                           useOctaves = use_octaves,
                                           nPixels = n_pixels,
                                           harmonize = harmony
                                    )
    
    return song, song_df, harmony

# Making the required prediction
if img2load is not None:
    # Saves
    img = Image.open(img2load)
    img = img.save("img.jpg")
    
    # OpenCv Read
    img = cv2.imread("img.jpg")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    #Display the image
    #st.image(img)
    
    idea , btn, _ = st.columns([2, 2, 1])

    if idea.button('Idea') == True:
        st.write("""
        The basic idea is as follows:

          1.  Images are made of pixels.
          2.  Pixels are composed of arrays of numbers that designate color
          3.  Color is described via color spaces like RGB or HSV for example
          4.  The color could be potentially mapped into a wavelength
          5.  Wavelength can be readily converted into a frequency
          6.  Sound is vibration that can be chracterized by frequencies
          7.  Therefore, an image could be translated into sound
        
        """)

    if btn.button('Create Music') == True:
        # generate music
        song, song_df, harmony = generate_music(img, 
                                                octave, 
                                                key, 
                                                scale, 
                                                sr_value, 
                                                t_value, 
                                                random_pixels, 
                                                use_octaves, 
                                                n_pixels, 
                                                harmony
                                        )

        progress = st.progress(0)
        for i in range(100):
            time.sleep(0.01)
            progress.progress(i+1)

        
        #Write the song into a file
        song_combined = np.vstack((song, harmony))
        wavfile.write('generated_song.wav', rate = 22050, data = song_combined.T.astype(np.float32))
        audio_file = open('generated_song.wav', 'rb')
        audio_bytes = audio_file.read()
        #Play the processed song
        st.audio(audio_bytes, format='audio/wav')

        # wavfile.write('simple_song.wav', rate = 22050, data = song.T.astype(np.float32))
        # audio_file1 = open('simple_song.wav', 'rb')
        # audio_bytes1 = audio_file1.read()
        # #Play the processed song
        # st.audio(audio_bytes1, format='audio/wav')

        
        
        #@st.cache
        def convert_df_to_csv(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode('utf-8')
        #csv = song_df.to_csv('song.csv')
        st.download_button('Download Song as CSV', data=convert_df_to_csv(song_df), file_name="song.csv",mime='text/csv',key='download-csv')

        st.balloons()

# While no image is uploaded
else:
    st.write("Waiting for an image to be uploaded...")
st.markdown("# Main page 🎈")
st.sidebar.markdown("# Main page 🎈")