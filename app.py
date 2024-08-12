import tempfile
import cv2
import streamlit as st
from ultralytics import YOLO
from streamlit_option_menu import option_menu
img="D:\WasteWise\Assets\Art.jpg"

def main():

    #Functions

    def home_page():
        
        st.markdown(
        """
        <style>
            .big-font {
                font-size:60px !important;
                line-height:1.3;
                padding-bottom:15px;
                padding-left:15px;
            }
            .medium-font {
                font-size:25px !important;
                color:grey;
                line-height:1.3;
                padding-bottom:15px;
                padding-left:15px;
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
            st.markdown('<button class="but">Get Started</button>', unsafe_allow_html=True)
        with col2:
            st.image(img,width=400)
    
    def test_page():
        st.title("Test")
        st.write("Welcome to the Test page of the app!")

        #Title

        st.title('WasteWise')

        #Sidebar-------------------------------------------------------------

        st.sidebar.title('Model')
        assigned_model_id = []
        assigned_model = st.sidebar.selectbox('Select The Model', list(model))
        assigned_model_id.append(model.index(assigned_model))

        st.sidebar.markdown('---')
        st.sidebar.title('Settings')

        st.markdown(
            """
            <style>
            [data-testid="stSidebar"][aria-expanded="true]> div:fist-child{width: 400px;}
            [data-testid="stSidebar"][aria-expanded="true]> div:fist-child{width: 400px; margin-left: -400px}
            </style>
            """,
            unsafe_allow_html = True,
        )
        
        confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)

        save_img = st.sidebar.checkbox('Save Output')
        enable_GPU = st.sidebar.checkbox('Enable GPU')
        custom_classes = st.sidebar.checkbox('Use Custom Classes')
        
        assigned_class_id = []
        if custom_classes:
            assigned_class = st.sidebar.multiselect('Select The Custom Classes', list(names), default='Plastic')
            for each in assigned_class:
                assigned_class_id.append(names.index(each))
        
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
        #Model

        trashModel = YOLO('Models/garbClass_25epochs.pt')
        waterModel = YOLO('Models/waterTrash_25epochs.pt')

        def predictTrash(source, save_img, confidence):
            prediction = trashModel.predict(source=source, save=save_img, conf=confidence)
            return prediction

        def count(results):
            for result in results:
                box = result.boxes
            return len(box)
        
        # Main Page

        DEMO_IMG = 'D:/WasteWise/Assets/trash_bottle.jpg'
        result = predictTrash(DEMO_IMG, save_img, confidence)
        n = count(result)
        st.markdown(n)

    def about_page():
        st.title("About")
        st.write("Welcome to the About page of the app!")

    def contact_page():
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
    model = ['Garbage Detection', 'Water Trash Detection']

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
