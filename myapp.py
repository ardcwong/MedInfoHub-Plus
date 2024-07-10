import streamlit as st


st.set_page_config(layout='wide')



# col2.write("")

# # col2.write("MedInfoHub is a comprehensive healthcare app designed to provide accessible medical information to patients and healthcare providers. Leveraging the power of the MedQuAD dataset* and advanced AI, MedInfoHub offers reliable answers to medical questions, supports telemedicine consultations, and enhances public health literacy. Whether you’re a patient seeking to understand your health better or a healthcare provider in need of quick, reliable information, MedInfoHub is your go-to resource for trusted medical knowledge.")
# col2.write("*Do you agree that ")
# col2.write("Press the 'Activate MedInfoHub' Button to begin exploring MedInfoHub.")

if "role" not in st.session_state:
    st.session_state.role = None

ROLES = ["", "Patient", "Health Care Provider", "Neither"]


def login():
    col1, col2, col3 = st.columns([1,3,1])
    
    col2.image('data/MIHv2.png')
    # col1.image('data/art.png')
    # st.header("Log in")
    col2.markdown('<p style="font-size: 18px; color: red;"><strong>⚠️ This app is not intended for self-diagnosis or self-treatment. Always consult a qualified healthcare professional for medical advice and diagnosis. ⚠️</strong></p>', unsafe_allow_html=True)
    
    role = col2.selectbox("Choose your role", ROLES)

    if col2.button("Enter MedInfoHub+"):
        st.session_state.role = role
        st.rerun()


def logout():
    st.session_state.role = None
    st.rerun()

def contactus():
    st.title('MedInfoHub')
    # st.subheader("WHAT WE OFFER")
    # st.image('data/use.png')
    st.subheader("CONTACT US")
    st.write('For any concerns or suggestions, you may reach out to us through the following:')
    contactinfo = """
    Facebook: facebook.com/medinfohub
    Twitter: twitter.com/medinfohub
    Instagram: instagram.com/medinfohub
    """
    # Display formatted text with st.markdown
    st.markdown(contactinfo, unsafe_allow_html=True)
def medinfohubplus():
    st.title('MedInfoHub+')
    # st.subheader("WHAT WE OFFER")
    # st.image('data/use.png')
    # st.subheader("CONTACT US")
    # st.write('For any concerns or suggestions, you may reach out to us through the following:')
    # contactinfo = """
    # Facebook: facebook.com/medinfohub
    # Twitter: twitter.com/medinfohub
    # Instagram: instagram.com/medinfohub
    # """
    # # Display formatted text with st.markdown
    # st.markdown(contactinfo, unsafe_allow_html=True)
    
role = st.session_state.role

logout_page = st.Page(logout, title="End Session", icon=":material/logout:")
about_us = st.Page(contactus, title="Contact Us", icon="✉️")
medinfohubplus_info = st.Page(medinfohubplus, title="About Our Data App", icon="📱", default=(role == role))


# settings = st.Page("settings.py", title="Settings", icon=":material/settings:")
medquad = st.Page(
    "MedQuAd/medquad.py",
    title="HealthPlus",
    icon="⚕️",
)
fda_app = st.Page(
    "FDA/fda_app.py", title="PharmaPal", icon="➕"
)

# 👩🏻‍⚕️
# respond_1 = st.Page(
#     "respond/respond_1.py",
#     title="Respond 1",
#     icon=":material/healing:",
#     default=(role == "Responder"),
# )
# respond_2 = st.Page(
#     "respond/respond_2.py", title="Respond 2", icon=":material/handyman:"
# )


# admin_1 = st.Page(
#     "admin/admin_1.py",
#     title="Admin 1",
#     icon=":material/person_add:",
#     default=(role == "Admin"),
# )
# admin_2 = st.Page("admin/admin_2.py", title="Admin 2", icon=":material/security:")
about_us_pages = [medinfohubplus_info,about_us]
account_pages = [logout_page]
data_apps = [medquad, fda_app]

# respond_pages = [respond_1, respond_2]
# admin_pages = [admin_1, admin_2]

# st.sidebar.title('MedInfoHub')
# with st.sidebar:
#     # st.subheader("WHAT WE OFFER")
#     # st.image('data/use.png')
#     st.subheader("CONTACT US")
#     st.write('For any concerns or suggestions, you may reach out to us through the following:')
#     contactinfo = """
#     Facebook: facebook.com/medinfohub
#     Twitter: twitter.com/medinfohub
#     Instagram: instagram.com/medinfohub
#     """
#     # Display formatted text with st.markdown
#     st.markdown(contactinfo, unsafe_allow_html=True)
# st.title("Request manager")
# st.logo("images/horizontal_blue.png", icon_image="images/icon_blue.png")

page_dict = {}

if st.session_state.role in ["Patient", "Health Care Provider", "Neither"]:
    page_dict["Application"] = data_apps
if st.session_state.role in ["Patient", "Health Care Provider", "Neither"]:
    page_dict["MedInfoHub+"] = about_us_pages

# if st.session_state.role in ["Responder", "Admin"]:
#     page_dict["Respond"] = respond_pages
# if st.session_state.role == "Admin":
#     page_dict["Admin"] = admin_pages

if len(page_dict) > 0:
    pg = st.navigation({"Session": account_pages} | page_dict)
else:
    pg = st.navigation([st.Page(login)]) #defaults to login page if no acceptable role is selected

pg.run()
