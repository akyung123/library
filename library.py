import streamlit as st
import pandas as pd
from streamlit_folium import folium_static
import folium
from folium.plugins import MarkerCluster
from sklearn.neighbors import NearestNeighbors
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import TruncatedSVD


# CSV 파일을 불러오는 함수
def load_data(file_path, nrows=None):
    return pd.read_csv(file_path, nrows=nrows)


# 도서 추천 함수 (NearestNeighbors와 TruncatedSVD 사용)
def recommend_books(title, data, n_components=100):
    # 입력받은 도서 제목에 해당하는 책 찾기
    book = data[data["TITLE_NM"].str.contains(title, case=False, na=False)]
    if book.empty:
        return None, None

    # ISBN_ADITION 벡터화
    count_vectorizer = CountVectorizer(analyzer="char", ngram_range=(1, 3))
    isbn_matrix = count_vectorizer.fit_transform(
        data["SGVL_ISBN_ADTION_SMBL_NM"].astype(str)
    )

    # Truncated SVD로 차원 축소
    svd = TruncatedSVD(n_components=n_components, random_state=42)
    isbn_reduced = svd.fit_transform(isbn_matrix)

    # NearestNeighbors로 유사한 책 찾기
    nn = NearestNeighbors(n_neighbors=6, metric="cosine", algorithm="brute")
    nn.fit(isbn_reduced)

    # 입력받은 책의 인덱스
    book_idx = book.index[0]

    # 유사한 책들의 인덱스 추출
    distances, indices = nn.kneighbors([isbn_reduced[book_idx]])
    similar_indices = indices.flatten()[1:]  # 입력 도서 제외

    # 추천 도서 5권 추출
    recommended_books = data.iloc[similar_indices]

    return book, recommended_books


# Session state 초기화
if "page" not in st.session_state:
    st.session_state.page = "start"

# flex-direction: column;  /* 세로 방향으로 배치 */
#             justify-content: center;  /* 수직 중앙 정렬 */
#             align-items: center;  /* 수평 중앙 정렬 */

# 첫 번째 페이지 (시작 페이지)
if st.session_state.page == "start":
    # JavaScript에서 이벤트를 Streamlit에 전달하는 HTML 코드
    # https://cdn.pixabay.com/photo/2024/07/26/19/48/book-8924228_960_720.png
    # https://cdn.pixabay.com/photo/2020/10/24/19/42/books-5682442_1280.jpg
    # https://cdn.pixabay.com/photo/2017/12/25/19/18/girl-3038974_960_720.jpg
    # https://cdn.pixabay.com/photo/2016/10/25/18/18/book-1769625_1280.png
    st.markdown(
        """
        <style>
        .start-page {
            background-image: url('https://cdn.pixabay.com/photo/2020/12/27/14/13/books-5864106_1280.jpg');
            background-size: contain;
            background-position: center;
            width: 50vw;
            height: 50vh;
            display: flex;
            flex-direction: column; /* 세로 방향으로 배치 */
            justify-content: space-between; /* 위아래 여백을 균등하게 분배 */
            align-items: center; /* 수평 중앙 정렬 */
            color: black;
            font-size: 60px;
        }

        .header-text {
            position: relative;
            top: 0;
            /* bottom: 5%;  하단에 고정 */
            left: 50%;
            text-align: center;
            color: beige;
            font-size: 60px;
            font-weight: bold;
            text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.3);  /* 텍스트 쉐도우 추가 */
        }

        .stButton button {
            background-color: beige;
            color: brown;
            padding: 10px 20px;
            font-size: 20px;
            font-weight: bold;
            border-radius: 8px;
            border: none;
            margin-top: 20px
            position: absolute;
            top: 0;
            width: 100%;
        }

        .button-container {
            display: flex;
            /*bottom: 10%;  버튼을 화면 중간 아래쪽에 배치 */
            justify-content: center;
            magint-top; 20px;
            text-align: center;  /* 버튼 내부도 중앙 정렬 */
        }

        </style>
        <div class="start-page">
            <div>
                <h1>📚 책책책</h1>
                <p>책의 위치를 알려드리고, 유사 도서를 추천해드립니다😍</p>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # 버튼을 중앙 아래에 배치
    st.markdown('<div class="button-container">', unsafe_allow_html=True)
    if st.button("책을 찾아서", key="start_button"):
        st.session_state.page = "main"
    st.markdown("</div>", unsafe_allow_html=True)

# 두 번째 페이지 (도서 검색 및 추천 기능)
elif st.session_state.page == "main":
    # 상단에 이전 페이지로 가는 버튼 추가
    if st.button("이전으로 가기", key="back_button"):
        st.session_state.page = "start"

    st.title("책책책 📚 책의 위치를 알려드리고, 유사 도서를 추천해드립니다😍")
    st.image('bb.png',use_column_width = True)
st.header(":rainbow[책책책]📚 책의 :blue[위치]를 알려드리고,:orange[유사도서]를 추천해드립니다😍", divider="rainbow")


    # 도서관 데이터 로드
    library_df = pd.read_csv("LIBRARY_202409.csv")  # 도서관 데이터 로드
    sido_options = library_df["ONE_AREA_NM"].unique()  # 시도 목록 추출

    # 도서 데이터 로드
    data = load_data("BOOK_PUB_202408.csv")

    # 지역 선택 (시도와 시군구)
    selected_sido = st.selectbox("시도를 선택하세요", ["전체"] + list(sido_options))

    if selected_sido != "전체":
        selected_sigungu = library_df[library_df["ONE_AREA_NM"] == selected_sido][
            "TWO_AREA_NM"
        ].unique()
        selected_sigungu = st.selectbox(
            "시군구를 선택하세요", ["전체"] + list(selected_sigungu)
        )

    # 도서 제목 입력
    title_input = st.text_input("도서 제목을 입력하세요")

    # 입력한 도서 제목과 유사한 책 목록 표시
    if title_input:
        # 입력한 제목과 유사한 모든 책 필터링
        book_matches = data[
            data["TITLE_NM"].str.contains(title_input, case=False, na=False)
        ]

        if not book_matches.empty:
            # 검색한 모든 책의 도서관 코드 목록 추출
            library_codes = book_matches["LBRRY_CD"].unique()

            # 해당 책들을 소장하고 있는 도서관 목록 필터링
            matched_libraries = library_df[library_df["LBRRY_CD"].isin(library_codes)]

            # 지역 필터링 적용
            if selected_sido != "전체":
                matched_libraries = matched_libraries[
                    matched_libraries["ONE_AREA_NM"] == selected_sido
                ]
                if selected_sigungu != "전체":
                    matched_libraries = matched_libraries[
                        matched_libraries["TWO_AREA_NM"] == selected_sigungu
                    ]

            # 소장 도서관 목록을 표로 표시
            st.write("### 소장 도서관 목록")
            st.dataframe(matched_libraries[["LBRRY_NM", "LBRRY_ADDR", "TEL_NO"]])

            # 지도 초기화 (임의의 시작 좌표, 예: 서울 기준)
            start_location = [35.157180, 129.062966]
            m = folium.Map(location=start_location, zoom_start=14)
            marker_cluster = MarkerCluster().add_to(m)

            # 도서관 마커 추가
            for i, j, name, addr, tel in zip(
                matched_libraries["LBRRY_LA"],
                matched_libraries["LBRRY_LO"],
                matched_libraries["LBRRY_NM"],
                matched_libraries["LBRRY_ADDR"],
                matched_libraries["TEL_NO"],
            ):
                marker = folium.CircleMarker(
                    location=[i, j],
                    radius=10,
                    color="green",
                    fill=True,
                    fill_color="green",
                    fill_opacity=0.6,
                    tooltip=f"<span style='font-size: 20px;'>{name}</span>",
                )

                # 팝업 추가
                popup_content = (
                    f"<div style='width: 200px; height: 100px; font-size: 18px;'>"
                    f"<strong>{name}</strong><br>주소: {addr}<br>전화번호: {tel}"
                    f"</div>"
                )

                folium.Popup(popup_content).add_to(marker)  # 팝업을 마커에 추가
                marker.add_to(marker_cluster)  # 마커 클러스터에 추가

            # 지도 표시
            folium_static(m)

            # 도서관 선택 옵션
            st.write("### 소장 도서관 선택")
            library_selected = st.selectbox(
                "도서관 선택", options=matched_libraries["LBRRY_NM"].tolist()
            )

            # 선택한 도서관의 소장 도서 현황 표시
            if library_selected:
                selected_library_code = matched_libraries[
                    matched_libraries["LBRRY_NM"] == library_selected
                ]["LBRRY_CD"].values[0]
                library_books = book_matches[
                    book_matches["LBRRY_CD"] == selected_library_code
                ]

                st.write("### 소장 도서 현황")
                st.dataframe(library_books[["TITLE_NM", "AUTHR_NM", "PBLICTE_YEAR"]])

            # 유사 도서 추천 표시
            st.write("### 유사 도서 추천")
            _, recommended_books = recommend_books(title_input, data)
            if recommended_books is not None and not recommended_books.empty:
                st.dataframe(recommended_books[["TITLE_NM", "AUTHR_NM", "PBLICTE_YEAR"]])
            else:
                st.write("추천할 유사 도서를 찾을 수 없습니다.")
        else:
            st.write("해당 도서 제목에 해당하는 책을 찾을 수 없습니다.")
