import streamlit_mermaid as stmd
import streamlit as st
import PyPDF2  # For reading PDF files
from openai import OpenAI
import re
import os
from dotenv import load_dotenv

load_dotenv()
openai_api_key = os.getenv("OPENAI_API_KEY")
client = OpenAI(api_key=openai_api_key)

# Function to generate HTML content for the Mermaid diagrams
def save_as_html(diagrams):
    html_content = """
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Mermaid Diagrams</title>
        <script type="module">
            import mermaid from 'https://cdn.jsdelivr.net/npm/mermaid@10/dist/mermaid.esm.min.mjs';
            mermaid.initialize({startOnLoad:true});
        </script>
    </head>
    <body>
    """
    
    for idx, diagram in enumerate(diagrams):
        html_content += f"""
        <div>
            <h2>Diagram {idx + 1}</h2>
            <div class="mermaid">
                {diagram}
            </div>
        </div>
        <br>
        """
    
    html_content += """
    </body>
    </html>
    """
    return html_content.encode('utf-8')

def extract_mermaid_diagrams(text):
    # 정규식을 사용하여 mermaid 코드 블록을 추출
    pattern = r'```mermaid\s*(.*?)\s*```'
    diagrams = re.findall(pattern, text, re.DOTALL)
    return diagrams

# Function to read text from a PDF file
def read_pdf(file):
    pdf_reader = PyPDF2.PdfReader(file)
    text = ""
    for page in pdf_reader.pages:
        text += page.extract_text()
    return text

def get_summary(text):
    system_message = """
    You are an expert summarizer. Your task is to provide a concise (Sentences + bullet points) and informative
    summary of the given text.
    
    Focus on the main ideas, key points, and essential information.
    
    Ensure that your summary is coherent, well-structured, and captures the essence of the original text.
    
    Aim for a summary that is approximately 10-15% of the length of the original text, unless the text is very
    short or long.
    """

    chat_completion = client.chat.completions.create(
      model="gpt-4o",  # Consider using the latest model that best fits your needs
      messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Please provide a comprehensive summary of the following text: {text}"}
      ]
    )
    response = chat_completion.choices[0].message.content

    return response

def get_mermaid(text):
    system_message = """[part 1]
syntax 에러 안나게 주의 해 주고, 구체적으로 들어갔던 전문적인 연구 디테일이 모두 반영되면 좋겠어. Mermaid 다이어그램에서 syntax 오류가 발생하는 이유는 주로 다음과 같으니 안나게 주의해줘. 

- 특수문자: Mermaid는 특수문자(예: (), @, &, $)를 변수나 노드 이름에 사용할 때 에러를 발생시킬 수 있습니다.
노드 간격 문제: Mermaid는 노드 이름 사이에 공백이 있을 때 문제가 생길 수 있습니다. 이 경우 따옴표나 백틱을 이용해 해결할 수 있습니다.
- subgraph 구문 문제: subgraph를 열고 닫을 때 반드시 올바르게 정리해야 하며, 그래프의 방향을 명확히 설정해야 합니다.
- 화살표 사용법: Mermaid는 --> (실선)과 -.-> (점선)를 구분합니다. 이 구분이 명확하지 않으면 오류가 발생할 수 있습니다.

이제, 첫 번째 답변과 두 번째 답변을 기준으로 mermaid 다이어그램을 그려줘. 즉 claim-evidence 그리고 imrd 관련 답변을 모두 한 번에 다이어그램화 시킬 거야. 그리고 각 노드별로 evidence 또는 imrd 태그를 붙여줘. 아래의 template을 따라줘.

graph TB
    %% Define Colors %%
    classDef question fill:#EF9A9A,stroke:#B71C1C,stroke-width:2px,color:#000;
    classDef solution fill:#BBDEFB,stroke:#0D47A1,stroke-width:2px,color:#000;
    classDef method fill:#C8E6C9,stroke:#1B5E20,stroke-width:2px,color:#000;
    classDef result fill:#FFF59D,stroke:#F57F17,stroke-width:2px,color:#000;
    classDef discussion fill:#FFCCBC,stroke:#BF360C,stroke-width:2px,color:#000;
    classDef conclusion fill:#C5E1A5,stroke:#33691E,stroke-width:2px,color:#000;
    classDef link fill:#FFECB3,stroke:#FF6F00,stroke-width:2px,color:#000;

    %% Introduction: Research Question and Proposed Solution %%
    subgraph Introduction
    I1["연구 문제: <연구 문제 1>"]:::question
    I2["연구 문제: <연구 문제 2>"]:::question
    I3["연구 목표: <연구 목표>"]:::question
    S1["제안된 솔루션: <솔루션 1>"]:::solution
    S2["제안된 솔루션: <솔루션 2>"]:::solution
    end

    %% Method: Experimental Design and Data %%
    subgraph Method
    M1["실험 방법: <실험 방법 1>"]:::method
    M2["실험 방법: <실험 방법 2>"]:::method
    M3["데이터 수집: <데이터 설명>"]:::method
    M4["실험 설계: <실험 설계 설명>"]:::method
    end

    %% Results: Key Findings and Relationships %%
    subgraph Results
    R1["결과: <결과 1>"]:::result
    R2["결과: <결과 2>"]:::result
    R3["결과: <결과 3>"]:::result
    R4["결과: <결과 4>"]:::result
    end

    %% Discussion: Contributions and Limitations %%
    subgraph Discussion
    D1["기여: <기여 1>"]:::discussion
    D2["기여: <기여 2>"]:::discussion
    L1["한계: <한계 1>"]:::discussion
    L2["한계: <한계 2>"]:::discussion
    end

    %% Conclusion: Future Directions %%
    subgraph Conclusion
    C1["향후 연구: <향후 연구 방향 1>"]:::conclusion
    C2["향후 연구: <향후 연구 방향 2>"]:::conclusion
    end

    %% Interconnections %%
    I1 --> S1
    I2 --> S2
    I3 --> S1

    S1 --> M1
    S2 --> M2
    M1 --> M3
    M2 --> M4

    M3 --> R1
    M3 --> R2
    M4 --> R3
    M4 --> R4

    R1 --> D1
    R2 --> D2
    R3 --> D1
    R4 --> D2

    D1 —> C1
    D2 —> C2
    L1 —> C1
    L2 —> C2

[part 2]
앞에 그래프를 만든 다음, 계속 이어서 part2에서는 앞의 그래프를 기반으로 subgraph를 꼭 만들어줘.  

목표:
1차 그래프의 핵심 노드(예: GAN 기반 모델, 품질 인식 손실 등)를 중심으로 각 섹션에서 제기된 주장과 근거가 연결되도록 합니다.
Key-Claim-Evidence 구조를 동적으로 연결하고, 각 세부 사항이 어떻게 연구의 주요 요소와 연결되는지 보여줘.
더 다이내믹한 레이아웃을 적용하여, key-주장-근거가 시간의 흐름에 따라 자연스럽게 상호 연결되도록 구성해봐.

각 섹션을 subgraph로 구분하여, Introduction, Method, Results, Discussion, Conclusion을 시각적으로 구획해.
그리고 각 섹션별로 따로따로 다이어그램 코드를 나눠서 제공해줘. 총 5개의 다이어그램 코드를 줘야겠지.

Step 1: Claim-Evidence 구조 생성
이제 각 섹션에 해당하는 Mermaid 다이어그램 코드를 각 섹션별로 더 자세한 내용을 반영하여 claim-evidence 구조로 변환해려고 해.
claim-evidence 구조를 활용하여 섹션 안의 모든 각 문장을 claim(주장)과 evidence(근거)로 세밀하게 나누어봐.
예를 들어, "시각적 정보 과부하는 사용자의 인지적 부담을 증가시킨다"라는 주장은, "UGC의 양이 기하급수적으로 증가하며 사용자가 이를 처리하기 어려워짐"이라는 근거로 뒷받침되어야겠지

Step 2: 
이제 1차 그래프의 모든 노드들과 claim-evidence 구조를 연결시켜야 해. 
우선 1차 그래프에서 나온 동일 섹션의 모든 노드들을 key 노드로 설정해. 예를 들어 introduction에 있던 노드가 4개라면, introdcution의 subgraph의 key 노드는 4개가 되어야 겠지.
각 key 노드에 따른 주장과 근거를 연결한 다이어그램을 만들어봐. 

claim, evidence 노드: 각 섹션 안에서는 구획을 나누지 않고 자유로운 레이아웃을 사용하여, 각 key 노드와 Claim-Evidence 구조를 동적으로 연결하는 그래프를 그려줘. 이 방식은 시간의 흐름을 따르면서도, 각 노드가 서로 유기적으로 연결될 수 있도록 설계해야해.

key 노드 간의 연결은 직선으로, Claim-Evidence 노드 연결은 실선으로 표시되어 전체적인 관계를 시각적으로 명확하게 보여줘.
시간 흐름: 논문에서 제시된 연구의 시간 흐름을 따르며, 각 단계가 자연스럽게 이어지도록 해

논문에 있는 문장 구조를 최대한 반영하고, 각 세트 간의 연결을 통해 연구의 흐름을 강조해.

Introduction에서 Conclusion까지 모든 내용을 가능한 한 놓치지 않고 반영하려 노력해.
이 다이어그램은 논문의 전체 흐름을 시각적으로 명확하게 보여줄 수 있을 것이며, 연구자가 쉽게 논문의 핵심 내용을 파악할 수 있도록 도와줄 것입니다.

밑의 예시 템플릿을 참고해. 
graph TB
    %% Define Colors %%
    classDef keynode fill:#FFECB3,stroke:#FF6F00,stroke-width:2px,color:#000;
    classDef claim fill:#BBDEFB,stroke:#0D47A1,stroke-width:2px,color:#000;
    classDef evidence fill:#C8E6C9,stroke:#1B5E20,stroke-width:2px,color:#000;

    %% Result Claims and Evidence %%
    N1["결과: {}"]:::keynode
    C1["Claim: {} ($H)"]:::claim
    E1["Evidence: {} ($H)"]:::evidence
    N1 --> C1
    C1 --> E1

    N2["결과: {}"]:::keynode
    C2["Claim: {} ($M)"]:::claim
    E2["Evidence: {} ($M)"]:::evidence
    N2 --> C2
    C2 --> E2

    N3["결과: {}"]:::keynode
    C3["Claim: {} ($M)"]:::claim
    E3["Evidence: {} ($M)"]:::evidence
    N3 --> C3
    C3 --> E3

    N4["결과: {}"]:::keynode
    C4["Claim: {} ($H)"]:::claim
    E4["Evidence: {} ($M)"]:::evidence
    N4 --> C4
    C4 --> E4

    %% 연결 관계 %%
    N1 --> N2
    N2 --> N3
    N3 --> N4

    %% 시간의 흐름 %%
    N1 -.-> N2
    N2 -.-> N3
    N3 -.-> N4
"""

    chat_completion = client.chat.completions.create(
      model="gpt-4o-mini",  # Consider using the latest model that best fits your needs
      messages=[
        {"role": "system", "content": system_message},
        {"role": "user", "content": f"Please provide the Mermaid Diagram code of the following text: {text}"}
      ]
    )
    response = chat_completion.choices[0].message.content

    diagrams = extract_mermaid_diagrams(response)

    return diagrams

# Set up the Streamlit page
st.set_page_config(layout="wide")

st.title("PaperFlow")

# Allow user to choose between pasting text or uploading a file
input_method = st.radio("Choose input method:", ["Paste Text", "Upload File"])

if input_method == "Paste Text":
    text = st.text_area("Enter your text here:")
else:
    uploaded_file = st.file_uploader("Choose a file", type=["txt", "pdf"])
    if uploaded_file is not None:
        if uploaded_file.type == "text/plain":
            text = uploaded_file.getvalue().decode("utf-8")
        elif uploaded_file.type == "application/pdf":
            with st.spinner("Loading PDF..."):
                text = read_pdf(uploaded_file)
            st.success("File uploaded successfully!")
        else:
            text = ""

if st.button("Generate Graph"):
    diagrams = get_mermaid(text)
    for diagram in diagrams:
        if len(diagrams) == 1:
            st.subheader("Diagram", divider=True)
        else:
            st.subheader(["Introduction", "Method", "Result", "Discussion", "Conclusion", "Other"][diagrams.index(diagram)], divider=True)
        stmd.st_mermaid(diagram, height="600px")
    
    # Add button to download HTML file with the Mermaid diagrams
    html_content = save_as_html(diagrams)
    
    st.download_button("Download Diagrams as HTML", html_content, file_name="diagrams.html", mime="text/html")
    