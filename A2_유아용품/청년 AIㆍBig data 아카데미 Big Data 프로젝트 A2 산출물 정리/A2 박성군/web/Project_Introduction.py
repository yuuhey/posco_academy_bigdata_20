import streamlit as st
#from multiapp import MultiApp
import utils
from PIL import Image


def app():
    st.set_page_config(
        page_title="유아용품 고객 분석과 프로모션 전략을 통한 고객 확보 및 매출 증진",
        page_icon="👋",
        layout="wide",
        initial_sidebar_state="expanded",
        menu_items={
            "Get help": "https://github.com/Lelp27/posco-service/bigdata",
            "Report a Bug": None,
            "About": "Posco 청년 AI·Big Data 20기 A2조",
        },
    )

    st.title("Posco Edu 20th A2")
    image = Image.open('/home/piai/workspace/posco-service/bigdata/data/posco-logo.png')
    st.image(image)
    """
    > 안녕하세요 With Posco A2조   
    > 장동언, 박성군, 전하영, 김민지, 전예찬, 이경로 입니다.
    
    ---
    
    `View PDF`를 통해서 발표 슬라이드를 보실 수 있습니다.  
    """
    st.sidebar.title("Posco Edu 20th A2")
    st.sidebar.write(
    """
    안녕하세요 Posco AI & Big data 아카데미 20기 A2조 입니다.  
    BigData Project 관련 데이터를 모아 시연용 Web App을 제작했습니다.
    
    Page 1 : Introduce & Presentation  
    Page 2 : Model 시연
    
    ### Source Code
    [Github](https://github.com/Lelp27/posco-service/tree/main/bigdata) : github.com/Lelp27

    """
    )

    st.download_button('⬇️ Download PDF', '/home/piai/workspace/posco-service/bigdata/data/presentation.pdf')
    with st.expander('View PDF on Web. \n 모바일에선 불가능 합니다..'):
        pdf_display = utils.show_pdf("/home/piai/workspace/posco-service/bigdata/data/presentation.pdf")
        st.markdown(pdf_display, unsafe_allow_html=True)
    image2 = Image.open('/home/piai/workspace/posco-service/bigdata/data/presentation_front.png')
    st.image(image2)


if __name__ == "__main__":
    app()