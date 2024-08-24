import json
import torch
import numpy as np
from matplotlib import pyplot as plt
from time import time
from PIL import Image 
import tempfile
import cv2
import streamlit as st
from ultralytics import YOLO
from streamlit_option_menu import option_menu

#Defining Assets
hero="https://i.pinimg.com/564x/a0/17/33/a01733a27004208af24df829129c08ab.jpg"
pfp="https://i.imgur.com/b25ZXQA.png"
demo_img="https://i.imgur.com/OdaZJSm.png"
high_confidence="https://i.imgur.com/HjpTdFB.png"
low_confidence="https://i.imgur.com/5uU0r6c.png"
trashNames = ['Biodegradable', 'Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic']
waterNames = ['Plastic','Bio','rov']
trashModel = YOLO('Models/garbClass_25epochs.pt')
waterModel = YOLO('Models/waterTrash_25epochs.pt')
sources=['Image','Video','Webcam']
model_list = {
    'Garbage Detection': trashModel,
    'Water Trash Detection': waterModel
}


def main():

    #Functions

    # def __init__(self, capture_index):
    #     self.capture_index = capture_index
    #     self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    #     print("Using Device: ", self.device)

    def predictTrash(model, source, save_img, confidence):
        if source.shape[2] == 4:
            source = cv2.cvtColor(source, cv2.COLOR_RGBA2RGB)
        prediction = model.predict(source=source, save=save_img, conf=confidence)

        detected_classes = []
        detected_confidences = []

        for result in prediction:
            if result.boxes is not None:
                for box in result.boxes:
                    label = model.names[int(box.cls[0])]
                    conf = box.conf[0].item()
                    detected_classes.append(label)
                    detected_confidences.append(conf)

        return prediction, detected_classes, detected_confidences

    def count(results):
        for result in results:
            box = result.boxes
        return len(box)

    def draw_bounding_boxes(image, results):
        for result in results:
            if result.boxes is not None:
                for box in result.boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist()) 
                    label = result.names[int(box.cls[0])]  
                    confidence = box.conf[0].item()

                    cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=5)
                    
                    cv2.putText(image, f"{label} {confidence:.2f}", (x1, y1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

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
            st.markdown('<p class="medium-font">Advanced Vision AI for Accurate Trash Detection and Streamlined Waste Management, ensuring a cleaner and greener future.</p>', unsafe_allow_html=True)

            # btn1=st.button(label="Get Started")
            st.markdown(
                """
                <div class="buttons-div">
                <button class="but" >Get Started</button>
                <button class="but" >Learn More</button>
                </div>
                """, 
                unsafe_allow_html=True
                )
            # if st.button('Test', key='test_button', on_click=lambda: navigate_to('Test')):
            #     pass
            # if st.button('About', key='about_button', on_click=lambda: navigate_to('About')):
            #     pass

        with col2:
            st.image(hero,width=400)

    def image(model, save_img, confidence):
        uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png", "webp"])
        if uploaded_file is not None:
            image = Image.open(uploaded_file)
            image = np.array(image)

            result, det_classes, det_confidences = predictTrash(model, image, save_img, confidence)
            det_obj_str = ", ".join(det_classes)  # Join list items with a comma
            det_conf_str = ", ".join(f"{conf:.3f}" for conf in det_confidences)  # Format confidence values

            image_with_boxes = draw_bounding_boxes(image, result)

            st.image(image_with_boxes, use_column_width=True)

            
            # Columns of info
            kpi1, kpi2, kpi3 = st.columns(3)

            with kpi1:
                n = count(result)
                st.markdown(
                    f"""
                    <div class="red-div">
                    <p class="big-font">Total Items</p>
                    <p class="medium-font">{n}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with kpi2:
                st.markdown(
                    f"""
                    <div class="grey-div">
                    <p class="big-font">Classes</p>
                    <p id="item-list" class="medium-font">{det_obj_str}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            with kpi3:
                st.markdown(
                    f"""
                    <div class="red-div">
                    <p class="big-font">Confidence</p>
                    <p class="medium-font">{det_conf_str}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    def video(model, save_img, confidence):

        video_file_buffer = st.file_uploader('Upload a video', type=['mp4', 'mov', 'avi', 'asf', 'm4v'])

        if video_file_buffer is not None:
            # Save the uploaded video to a temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as tmp_file:
                tmp_file.write(video_file_buffer.read())
                video_path = tmp_file.name

            # Open the video file using OpenCV
            cap = cv2.VideoCapture(video_path)

            # Retrieve video properties
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = cap.get(cv2.CAP_PROP_FPS)

            # Define the codec and create a VideoWriter object to save the processed video
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")  # Use 'XVID' for .avi files
            with tempfile.NamedTemporaryFile(delete=False, suffix=".mp4") as processed_file:
                processed_video_path = processed_file.name

            out = cv2.VideoWriter(processed_video_path, fourcc, fps, (width, height))

            st.info("Processing video... This may take a while depending on the video length.")

            # Process each frame of the video
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            progress_bar = st.progress(0)
            current_frame = 0

            
            
            # Set up KPI placeholders using Streamlit's empty containers
            kpi1_placeholder = st.empty()
            kpi2_placeholder = st.empty()
            kpi3_placeholder = st.empty()
            kpi4_placeholder = st.empty()

            # Initialize the KPI displays
            kpi1_placeholder.markdown(
                """
                <div class="red-div">
                    <p class="big-font">Total Items</p>
                    <p class="medium-font">0</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            kpi2_placeholder.markdown(
                """
                <div class="grey-div">
                    <p class="big-font">Classes</p>
                    <p class="medium-font">0</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            kpi3_placeholder.markdown(
                """
                <div class="red-div">
                    <p class="big-font">Confidence</p>
                    <p class="medium-font">0.0</p>
                </div>
                """,
                unsafe_allow_html=True
            )
            kpi4_placeholder.markdown(
                f"""
                <div class="grey-div">
                    <p class="big-font">Frame Rate</p>
                    <p class="medium-font">{fps}</p>
                </div>
                """,
                unsafe_allow_html=True
            )

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                #Progress bar
                current_frame += 1
                progress = current_frame / frame_count
                progress_bar.progress(min(progress, 1.0))


                # Perform prediction using the YOLO model
                results, det_obj, det_conf = predictTrash(model, frame, save_img, confidence)
                number = count(results)
                frame = draw_bounding_boxes(frame, results)

                # Convert lists to strings for display
                det_obj_str = ", ".join(det_obj)  # Join list items with a comma
                det_conf_str = ", ".join(f"{conf:.2f}" for conf in det_conf)  # Format confidence values


                # Write the processed frame to the output video
                out.write(frame)

                # Update KPI placeholders with the latest values
                kpi1_placeholder.markdown(
                    f"""
                    <div class="red-div">
                        <p class="big-font">Total Items</p>
                        <p class="medium-font">{number}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                kpi2_placeholder.markdown(
                    f"""
                    <div class="grey-div">
                        <p class="big-font">Classes</p>
                        <p class="medium-font">{det_obj_str}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                kpi3_placeholder.markdown(
                    f"""
                    <div class="red-div">
                        <p class="big-font">Confidence</p>
                        <p class="medium-font">{det_conf_str}</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )


            # Release resources
            cap.release()
            out.release()
            cv2.destroyAllWindows()

            # Display the processed video
            st.success("Processing complete!")
            st.video(processed_video_path)

        else:
            st.warning("Please upload a video file to proceed.")




    def webcam(model, save_img, confidence):
        # Columns of info
        kpi1, kpi2, kpi3, kpi4 = st.columns(4)

        with kpi1:
            st.markdown(
                """
                <div class="red-div">
                <p class="big-font">Total Items</p>
                <p class="medium-font">0</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with kpi2:
            st.markdown(
                """
                <div class="grey-div">
                <p class="big-font">Classes</p>
                <p class="medium-font">0</p>
                </div>
                """,
                unsafe_allow_html=True
            )

        with kpi3:
            st.markdown(
                """
                <div class="red-div">
                <p class="big-font">Confidence</p>
                <p class="medium-font">0</p>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with kpi4:
                st.markdown(
                    """
                    <div class="grey-div">
                    <p class="big-font">Frame Rate</p>
                    <p class="medium-font">0</p>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

    def test_page():
        
        #Sidebar-------------------------------------------------------------
        with st.sidebar:
            
            #Select Model
            st.title('Model')
            assigned_model_display_name = st.selectbox('Select The Model', list(model_list.keys()))
            model = model_list[assigned_model_display_name]

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
            
            #Settings - Confidence 
            confidence = st.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)

            #Settings - Checkboxes
            save_img = st.checkbox('Save Output')
            enable_GPU = st.checkbox('Enable GPU')
            custom_classes = st.checkbox('Use Custom Classes')

            if assigned_model_display_name == "Water Trash Detection":
                names=waterNames
            else :
                names=trashNames

            assigned_class_id = []
            if custom_classes:
                assigned_class = st.multiselect('Select The Custom Classes', list(names), default="Plastic")
                for each in assigned_class:
                    assigned_class_id.append(names.index(each))


        #Main Page-----------------------------------------------------------

        st.markdown("")

        #Select the medium
        assigned_source = st.selectbox('Select The Source', list(sources))

        #Upload file

        #Upload file - Styles
        st.markdown(
            """
            <style>
                .red-div{
                    background-color: #FF4B4B;
                    min-height: 30vh;
                    display:flex;
                    flex-direction: column;
                    align-items:center;
                    justify-content:center;
                    color: #ECECEC;
                    border-radius:25px;
                    padding: 20px;
                }
                .grey-div{
                    background-color: #D6D6D6;
                    min-height: 30vh;
                    display:flex;
                    flex-direction: column;
                    align-items:center;
                    justify-content:center;
                    border-radius:25px;
                    padding: 20px;
                }
                .big-font {
                    font-size:40px !important;
                    font-weight:bold;
                }
                .medium-font {
                    font-size:25px !important;
                    text-align:center;
                }
            </style>
            """,
            unsafe_allow_html=True
        )

        #Upload File - Image
        if assigned_source == "Image":
            image(model, save_img, confidence)
                
        #Upload File - Video     
        elif assigned_source == "Video":
            video(model, save_img, confidence)

        #Upload File - Webcam
        elif assigned_source == "Webcam":
            webcam(model, save_img, confidence)            
          
    def about_page():
        #Styles
        st.markdown(
        """
        <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons/font/bootstrap-icons.css" rel="stylesheet">
        <style>
            .wrapper{
                display: flex;
                flex-direction: column;
                justify-content: center;
                margin-top: 20px;
                margin-bottom: 50px;
                padding-left: 10px;
                padding-right: 10px;
            }
            .wrapper-center{
                display: flex;
                flex-direction: column;
                justify-content: center;
                margin-top: 20px;
                padding-left: 10px;
                padding-right: 10px;
                align-items:center;
                text-align:center;
            }
            .big-font {
                font-size:60px !important;
                line-height:1.1;
                padding-left:15px;
                margin-bottom:30px;
            }
            .sub-header-font{
                font-size:40px !important;
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
            .center-div{
                width:94vw;
                display: flex;
                justify-content: center;
            }
        </style>
        """,
        unsafe_allow_html=True,
        )

        #Introduction Paragraph
        st.markdown(
            """
            <div class="wrapper">
                <p class="big-font">Why is WasteWise essential for modern waste management?</p>
                <p class="medium-font">WasteWise is a crucial tool for modern waste management, offering an innovative solution to effective waste segregation. By accurately detecting and classifying waste, it simplifies the separation of recyclables from general waste, improving collection efficiency. This benefits the environment, enhances worker safety by reducing manual sorting errors, and integrates seamlessly into household waste management. Additionally, WasteWise aids in underwater environments by detecting and segregating aquatic waste, supporting water pollution control and marine ecosystem preservation. Overall, WasteWise plays a vital role in advancing sustainable waste management.</p>
            </div>
            """,
            unsafe_allow_html=True
        )

        #Demo Section
        st.markdown(
            """
            <div class="wrapper-center">
                <p class="big-font">See WasteWise in Action</p> 
            </div>
            """,
            unsafe_allow_html=True,
        )
        #Demo Section - Image
        st.markdown(
            f"""
            <p class="sub-header-font"><i class="bi bi-caret-right-fill"></i> Using Image</p>
            <div class="center-div">
                <img style="margin-top:20px;" src={demo_img} width="85%">
            </div>
            """,
            unsafe_allow_html=True
        )
        #Demo Section - Video
        #Demo Section - Webcam

        #Additional Settings
        st.markdown(
            """
            <div class="wrapper-center">
                <p class="big-font">Additional Settings</p> 
            </div>
            """,
            unsafe_allow_html=True,
            )

        #Additional Settings - Confidence section
        st.markdown(
            f"""
            <p class="sub-header-font"><i class="bi bi-caret-right-fill"></i> Changing the Confidence</p>
            <p class="medium-font" style="margin-left:50px;">
                You can adjust the model's confidence level in the settings section found in the sidebar.
            </p>
            <p class="medium-font" style="margin-left:50px; margin-bottom:30px;">
                Higher confidence increases the accuracy of object detection whereas Lower confidence allows the model to detect a larger number of objects.
            </p>
            <div style="width:94vw;
                        display: flex;
                        justify-content: flex-start; 
                        margin-bottom:30px;
                        padding-left:20px;
                        padding-right:20px;
                    ">
                <img src={high_confidence} width="65%">
            </div>
            <div style="width:94vw;
                        display: flex;
                        justify-content: flex-end; 
                        margin-bottom:30px;
                        margin-right:30px;
                        padding-right:40px;
                    ">
                <img src={low_confidence} width="65%">
            </div>
            """,
            unsafe_allow_html=True
        )
        
    def contact_page():
        #Styles
        st.markdown(
        """
        <link href="https://cdn.jsdelivr.net/npm/bootstrap-icons/font/bootstrap-icons.css" rel="stylesheet">
        <style>
            .wrapper{
                display: flex;
                flex-direction: column;
                justify-content: center;
                margin-top: 10px;
                padding-left: 10px;
                padding-right: 10px;
            }
            .big-font {
                font-size:60px !important;
                line-height:1.1;
                padding-left:15px;
                margin-bottom:40px;
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

        #Sidebar
        with st.sidebar:
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
            st.markdown(
                """
                <h1 style="text-align: right;"><i class="bi bi-arrow-down-left"></i></h1>
                <br></br>
                """,
                unsafe_allow_html=True,
            )

            #Contact Details
            st.markdown(
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
        st.markdown("""
                    <div class="wrapper">
                    <p class="big-font">Leveraging machine learning to convert data into innovative solutions.</p>
                    </div>
                    """,unsafe_allow_html=True)
        col1, col2 = st.columns([2,1],gap="medium",vertical_alignment="center")
        with col1:
            st.markdown("""
                        <div class="wrapper">
                        <p class="medium-font">I'm <b><u>Vedant Sudhir Patil</u></b>, a web developer and machine learning engineer.<br></br>
                                            I contributed to Social Summer of Code, Season 3, and specialize in building impactful projects using Python, JavaScript, Java, React.js and Node.js.
                        </p>
                        </div>
                        """,unsafe_allow_html=True)
        with col2:
            st.image(pfp)
    
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

    hide_hamburger_menu()

    #Navbar
    if 'selected_page' not in st.session_state:
        st.session_state.selected_page = 'Home'

    def navigate_to(page):
        st.session_state.selected_page = page

    selected = option_menu(
        menu_title=None,
        options=['Home', 'Test', 'About', 'Contact'],
        icons=['house', 'code-slash', 'book', 'envelope'],
        menu_icon='cast',
        default_index=0,
        orientation='horizontal',
    )
    st.session_state.selected_page = selected
    
    if st.session_state.selected_page == 'Test':
        test_page()  
    elif st.session_state.selected_page == 'About':
        about_page()
    elif st.session_state.selected_page == 'Contact':
        contact_page()
    else:
        home_page()

if __name__=='__main__':
    try:
        main()
    except SystemExit:
        pass
