
import tempfile
import cv2
import streamlit as st

def main():

    names = ['Biodegradable', 'Cardboard', 'Glass', 'Metal', 'Paper', 'Plastic']

    st.title('WasteWise')

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

    st.sidebar.markdown('---')
    confidence = st.sidebar.slider('Confidence', min_value=0.0, max_value=1.0, value=0.25)
    st.sidebar.markdown('---')

    save_img = st.sidebar.checkbox('Save Video')
    enable_GPU = st.sidebar.checkbox('Enable GPU')
    custom_classes = st.sidebar.checkbox('Use Custom Classes')
    assigned_class_id = []

    if custom_classes:
        assigned_class = st.sidebar.multiselect('Select The Custom Classes', list(names), default='Plastic')
        for each in assigned_class:
            assigned_class_id.append(names.index(each))
    
    video_file_buffer = st.sidebar.file_uploader('Upload a video', type= ['mp4', 'mov', 'avi', 'asf', 'm4v'])
    DEMO_VIDEO='D:/WasteWise/Assets/video.mp4'
    tffile = tempfile.NamedTemporaryFile(suffix='.mp4', delete=False)

    if not video_file_buffer:
        vid = cv2.VideoCapture(DEMO_VIDEO)
        tffile.name=DEMO_VIDEO
        dem_vid=open(tffile, 'rb')
        demo_bytes = dem_vid.read()

        st.sidebar.text('Input Video')
        st.sidebar.video(demo_bytes)
    else:
        tffile.write(video_file_buffer.read())
        dem_vid = open(tffile.name, 'rb')
        demo_bytes = dem_vid.read()

        st.sidebar.text('Input Video')
        st.sidebar.video(demo_bytes)
    print(tffile.name)

    stframe= st.empty()
    st.sidebar.markdown('---')

    kpi1, kpi2, kpi3 = st.columns(3)

    with kpi1:
        st.markdown("**Frame Rate**")
        kpi1_text= st.markdown(0)

    with kpi2:
        st.markdown("**Tracked Objects**")
        kpi1_text= st.markdown(0)

    with kpi3:
        st.markdown("**Width**")
        kpi1_text= st.markdown(0)




if __name__=='__main__':
    try:
        main()
    except SystemExit:
        pass
