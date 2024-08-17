import torch
import numpy as np
from time import time
from PIL import Image 
import tempfile
import cv2
import streamlit as st
from ultralytics import YOLO
from streamlit_option_menu import option_menu
img="https://i.pinimg.com/564x/a0/17/33/a01733a27004208af24df829129c08ab.jpg"

def main():

    #Functions

    # def __init__(self, capture_index):
    #     self.capture_index = capture_index
    #     self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     print("Using Device: ", self.device)

    def predictTrash(model, source, save_img, confidence):
        prediction = model.predict(source=source, save=save_img, conf=confidence)
        return prediction

    def count(results):
        for result in results:
            box = result.boxes
        return len(box)

    def draw_bounding_boxes(image, results):
        #Function to draw bounding boxes on the image based on the YOLO output.
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    # Extract the bounding box coordinates
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())  # Convert to integer
                    label = result.names[int(box.cls[0])]  # Get the label using the class index
                    confidence = box.conf[0].item()  # Get the confidence score
                    
                    # Draw the bounding box
                    cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)
                    
                    # Draw the label and confidence
                    cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

        return image
          
    def home_page():
        
        st.markdown(
        """
        <style>
            .big-font {
                font-size:60px !important;
                line-height:1.3;
                padding-left:15px;
            }
            .medium-font {
                font-size:25px !important;
                color:grey;
                line-height:1.3;
                padding-bottom:0px;
                padding-left:15px;
            }
            .buttons-div{
                display:flex;
                align-items:center;
                # justify-content:center;
                justify-content:flex-start;
                gap:1vw;
            }
            .but {
                margin-left:15px;
                text-align:center;
                border-radius:5px;
                padding:5px 10px 5px 10px;
                border: solid 1px #FF4B4B;
                color:#ECECEC;
                background-color:#FF4B4B;
            }
            .but:active {
                border: solid 1px #FF4B4B;
                background-color: white;
                color:#FF4B4B;
            }
        </style>
        """,
        unsafe_allow_html=True,
        )

        col1, col2 = st.columns([2,1],gap="medium",vertical_alignment="center")
        with col1:
            st.markdown('<p class="big-font">Empowering Waste Management with <b>Waste Wise</b></p>', unsafe_allow_html=True)
            st.markdown('<p class="medium-font">Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. </p>', unsafe_allow_html=True)
            #btn1=st.button(label="Get Started")
            st.markdown(
                """
                <div class="buttons-div">
                <button class="but">Get Started</button>
                <button class="but">Learn More</button>
                </div>
                """, 
                unsafe_allow_html=True
                )

        with col2:
            st.image(img,width=400)
    
    def test_page():
        
        #Sidebar-------------------------------------------------------------
        with st.sidebar:
            
            #Select Model
            st.title('Model')
            assigned_model_display_name = st.selectbox('Select The Model', list(model_list.keys()))
            model = model_list[assigned_model_display_name]
            assigned_model_id = list(model_list.keys()).index(assigned_model_display_name)

            st.markdown('---')

            #Settings
            st.title('Settings')

            st.markdown(
                """
                <style>
                [data-testid="stSidebar"][aria-expanded="true"]> div:fist-child{width: 400px;}
                [data-testid="stSidebar"][aria-expanded="true"]> div:fist-child{width: 400px; margin-left: -400px}
                </style>
                """,
                unsafe_allow_html = True,
            )
            
            confidence = st.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)

            save_img = st.checkbox('Save Output')
            enable_GPU = st.checkbox('Enable GPU')
            custom_classes = st.checkbox('Use Custom Classes')
            
            assigned_class_id = []
            if custom_classes:
                assigned_class = st.multiselect('Select The Custom Classes', list(names), default='Plastic')
                for each in assigned_class:
                    assigned_class_id.append(names.index(each))
            
            #start = st.button('Test Model')
            
        # video_file_buffer = st.sidebar.file_uploader('Upload a video', type= ['mp4', 'mov', 'avi', 'asf', 'm4v'])
        # DEMO_VIDEO='D:/WasteWise/Assets/video.mp4'
        # tffile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)

        # if not video_file_buffer:
        #     vid = cv2.VideoCapture(DEMO_VIDEO)
        #     tffile.name=DEMO_VIDEO
        #     dem_vid=open(tffile, 'rb')
        #     demo_bytes = dem_vid.read()

        #     st.sidebar.text('Input Video')
        #     st.sidebar.video(demo_bytes)
        # else:
        #     tffile.write(video_file_buffer.read())
        #     dem_vid = open(tffile.name, 'rb')
        #     demo_bytes = dem_vid.read()

        #     st.sidebar.text('Input Video')
        #     st.sidebar.video(demo_bytes)
        # print(tffile.name)

        # stframe= st.empty()
        # st.sidebar.markdown('---')

        # kpi1, kpi2, kpi3 = st.columns(3)

        # with kpi1:
        #     st.markdown("**Frame Rate**")
        #     kpi1_text= st.markdown(0)

        # with kpi2:
        #     st.markdown("**Tracked Objects**")
        #     kpi2_text= st.markdown(0)

        # with kpi3:
        #     st.markdown("**Width**")
        #     kpi3_text= st.markdown(0)

        #Main Page-----------------------------------------------------------

        #Select the medium
        assigned_source_id = []
        assigned_source = st.selectbox('Select The Source', list(sources))
        assigned_source_id.append(sources.index(assigned_source))

         #Upload file
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = np.array(image)

            result = predictTrash(model, image, save_img, confidence)

            image_with_boxes = draw_bounding_boxes(image, result)

            st.image(image_with_boxes, caption='Processed Image', use_column_width=True)
            
            n = count(result)
            st.markdown(f"Total objects detected: **{n}**")
        
    def about_page():
        st.markdown(
        """
        <style>
            .wrapper{
                display: flex;
                flex-direction: column;
                justify-content: center;
                margin-top: 20px;
                margin-bottom: 70px;
                padding-left: 10px;
                padding-right: 10px;
            }
            .wrapper-center{
                display: flex;
                flex-direction: column;
                justify-content: center;
                margin-top: 20px;
                margin-bottom: 70px;
                padding-left: 10px;
                padding-right: 10px;
                align-items:center;
                text-align:center;
            }
            .big-font {
                font-size:60px !important;
                line-height:1.1;
                padding-left:15px;
            }
            .medium-font {
                font-size:25px !important;
                color:grey;
                line-height:1.3;
                padding-bottom:0px;
                padding-left:15px;
            }
        </style>
        """,
        unsafe_allow_html=True,
        )

        st.markdown(
            """
            <div class="wrapper">
                <p class="big-font">Why is WasteWise essential for modern waste management?</p>
                <p class="medium-font">WasteWise is crucial for modern waste management as it provides an innovative solution to one of the most persistent challenges — effective waste segregation. By accurately detecting and classifying different types of waste, WasteWise simplifies the process of separating recyclable materials from general waste, making collection and disposal more efficient. This not only benefits the environment but also assists workers who struggle to identify and sort waste manually, reducing the margin for error and enhancing their safety. WasteWise can be seamlessly integrated into household waste management, ensuring that organic, recyclable, and hazardous materials are correctly sorted at the source, thereby streamlining the recycling process. Additionally, its application extends to underwater environments, where it aids in the detection and segregation of aquatic waste, a crucial step in combating water pollution and preserving marine ecosystems. By addressing these multifaceted issues, WasteWise emerges as an indispensable tool in our collective efforts toward sustainable waste management.</p>
            </div>
            <div class="wrapper-center">
                <p class="big-font">See WasteWise in Action</p> 
            </div>
            """,
            unsafe_allow_html=True,
            )

    def contact_page():
        #Assigning bootstap icons to the page
        st.markdown(
            """
            <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons/font/bootstrap-icons.css" rel="stylesheet">
            """,
            unsafe_allow_html=True,
        )

        #Sidebar------------------------
        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true"]>{width: 900px;}
            [data-testid="stSidebar"][aria-expanded="true"]>{width: 900px; margin-left: -400px}
            </style>
            """,
            unsafe_allow_html = True,
        )

        #Arrow icon
        st.sidebar.markdown(
            """
            <h1 style="text-align: right;"><i class="bi bi-arrow-down-left"></i></h1>
            <br></br>
            """,
            unsafe_allow_html=True,
        )

        #Contact Details
        st.sidebar.markdown(
            """
            <div>
            <h4 style="color: #FF4B4B">Contact Details</h4>
            <h2 style="line-height:0.9">
                <i class="bi bi-envelope-at"></i><a href="mailto:vedantsudhirpatil.com" style="text-decoration: none; color:#31333F; margin-left:5px;"> vedantsudhirpatil@gmail.com</a><br></br>
                <i class="bi bi-telephone"></i><a href="tel: +919540386773" style="text-decoration: none; color:#31333F; margin-left:5px;"> +91 9540386773</a>
            </h2>
            <br></br>
            <div>
            <div>
            <h4 style="color: #FF4B4B">Socials</h4>
            <h2 style="line-height:0.9">
                <i class="bi bi-linkedin"></i><a href="https://www.linkedin.com/in/vedantspatil" style="text-decoration: none; color:#31333F; margin-left:10px;"> Linkedin</a><br></br>
                <i class="bi bi-github"></i><a href="https://github.com/Vedant-SPatil" style="text-decoration: none; color:#31333F; margin-left:10px;"> Github</a><br></br>
                <i class="bi bi-twitter-x"></i><a href="https://twitter.com/Vedant_SPatil" style="text-decoration: none; color:#31333F; margin-left:10px;"> Twitter</a><br></br>
                <i class="bi bi-instagram"></i><a href="https://www.instagram.com/vedant_spatil/" style="text-decoration: none; color:#31333F; margin-left:10px;"> Instagram</a>
            </h2>
            <div>
            """,
            unsafe_allow_html=True,
        )

        #Main Page
        st.title("Contact")
        st.write("Welcome to the Contact page of the app!")
    
    def hide_hamburger_menu():
        hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
        st.markdown(hide_streamlit_style, unsafe_allow_html=True)


    ######      MAIN INFO      ######

    #Page Config
    st.set_page_config(
        page_title="WasteWise",
        page_icon=":trash-fill:",
        layout="wide",
    )

    #Streamlit (Information Arrays)
    names = ['Biodegradable', 'Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic']
    trashModel = YOLO('Models/garbClass_25epochs.pt')
    waterModel = YOLO('Models/waterTrash_25epochs.pt')
    sources=['Image','Video','Webcam']
    model_list = {
        'Garbage Detection': trashModel,
        'Water Trash Detection': waterModel
    }
    DEMO_IMG = 'D:/WasteWise/Assets/trash_bottle.jpg'


    #Navbar
    selected = option_menu(
        menu_title=None,
        options=['Home', 'Test', 'About', 'Contact'],
        icons=['house', 'code-slash', 'book', 'envelope'],
        menu_icon='cast',
        default_index=0,
        orientation='horizontal',
    )
    
    if selected == 'Test':
        test_page()  
    elif selected == 'About':
        about_page()
    elif selected == 'Contact':
        contact_page()
    else:
        home_page()

if __name__=='__main__':
    try:
        main()
    except SystemExit:
        pass
