{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/LuckyMan26/RAG/blob/main/app.py\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "s-ipqcqzkQuK"
      },
      "outputs": [],
      "source": [
        "!pip install git+https://github.com/tantanchen/dspy.git\n",
        "!pip install groq\n",
        "!pip install colbert"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "EW1rLWrbNzM5"
      },
      "outputs": [],
      "source": [
        "!pip install chromadb"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ixTwF9nWOG-M"
      },
      "outputs": [],
      "source": [
        "!pip install -qU \\\n",
        "    transformers==4.30.2 \\\n",
        "    torch==2.0.1 \\\n",
        "    einops==0.6.1 \\\n",
        "    accelerate==0.20.3 \\\n",
        "    datasets==2.14.5 \\\n",
        "    chromadb \\\n",
        "    sentence-transformers==2.2.2"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "hCyvBaQ4VEau"
      },
      "outputs": [],
      "source": [
        "!pip install langchain"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "YSmNHXPdHntM",
        "outputId": "3193e9dd-4047-4506-b957-55b76e2d737e"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\u001b[33mWARNING: Skipping lark as it is not installed.\u001b[0m\u001b[33m\n",
            "\u001b[0m"
          ]
        }
      ],
      "source": [
        "!pip uninstall lark --yes"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "EGKnsvvsJF9g"
      },
      "outputs": [],
      "source": [
        "!pip install lark-parser"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "TY6MUZayJV-W"
      },
      "outputs": [],
      "source": [
        "!pip install lark"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ZGH0QStGoyhy"
      },
      "outputs": [],
      "source": [
        "import dspy\n",
        "import groq\n",
        "import colbert"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 9,
      "metadata": {
        "id": "0jeuNz9WOBvF"
      },
      "outputs": [],
      "source": [
        "import chromadb\n",
        "from sentence_transformers import SentenceTransformer"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "d9MRE4IAYtuM"
      },
      "outputs": [],
      "source": [
        "from langchain.text_splitter import TextSplitter\n",
        "class CaseSplitter(TextSplitter):\n",
        "    def __init__(self):\n",
        "        super().__init__()\n",
        "\n",
        "    def split_text(self, file):\n",
        "\n",
        "        cases = file.strip().split(\"\\n\\nRow \")\n",
        "        return cases\n",
        "\n",
        "\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "1do9KxXLVC7x"
      },
      "outputs": [],
      "source": [
        "from langchain.text_splitter import RecursiveCharacterTextSplitter\n",
        "text_splitter = RecursiveCharacterTextSplitter(\n",
        "    chunk_size = 1500,\n",
        "    chunk_overlap = 150\n",
        ")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "EeVydcAIri95"
      },
      "outputs": [],
      "source": [
        "from google.colab import files\n",
        "files.upload()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9CnIpxOlrLIp"
      },
      "outputs": [],
      "source": [
        "from langchain.document_loaders import TextLoader\n",
        "loader = TextLoader(\"processed_csv.txt\", encoding=\"cp1252\")\n",
        "documents = loader.load()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "fRKX4TE1bBKf"
      },
      "outputs": [],
      "source": [
        "with open(\"processed_csv.txt\", encoding=\"cp1252\") as f:\n",
        "    state_of_the_union = f.read()\n",
        "case_splitter = CaseSplitter()\n",
        "cases = case_splitter.split_text(state_of_the_union)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 89
        },
        "id": "vseZJCmKhu4m",
        "outputId": "d2b510fa-6315-420c-9c4b-db66ede7e57f"
      },
      "outputs": [
        {
          "data": {
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            },
            "text/plain": [
              "'2:\\ncaseId: 1946-002\\ndocketId: 1946-002-01\\ncaseIssuesId: 1946-002-01-01\\nvoteId: 1946-002-01-01-01\\ndateDecision: 11/18/1946\\ndecisionType: opinion of the court (orally argued)\\nusCite: 329 U.S. 14\\nsctCite: 67 S. Ct. 13\\nledCite: 91 L. Ed. 12\\nlexisCite: 1946 U.S. LEXIS 1725\\nterm: 1946\\nnaturalCourt: Vinson 1 \\tJune 24, 1946 - August 23, 1949\\nchief: Vinson\\ndocket: 12\\ncaseName: CLEVELAND v. UNITED STATES\\ndateArgument: 10/10/1945\\ndateRearg: 10/17/1946\\npetitioner: person accused, indicted, or suspected of crime\\npetitionerState: \\nrespondent: United States\\nrespondentState: \\njurisdiction: cert\\nadminAction: \\nadminActionState: \\nthreeJudgeFdc: no mention that a 3-judge ct heard case\\ncaseOrigin: Utah U.S. District Court\\ncaseOriginState: 52\\ncaseSource: U.S. Court of Appeals, Tenth Circuit\\ncaseSourceState: \\nlcDisagreement: no mention that dissent occurred\\ncertReason: putative conflict\\nlcDisposition: affirmed\\nlcDispositionDirection: conservative\\ndeclarationUncon: no declaration of unconstitutionality\\ncaseDisposition: affirmed (includes modified)\\ncaseDispositionUnusual: no unusual disposition specified\\npartyWinning: no favorable disposition for petitioning party apparent\\nprecedentAlteration: no determinable alteration of precedent\\nvoteUnclear: vote clearly specified\\nissue: statutory construction of criminal laws: Mann Act and related statutes\\nissueArea: Criminal Procedure\\ndecisionDirection: conservative\\ndecisionDirectionDissent: dissent in opposite direction\\nauthorityDecision1: statutory construction\\nauthorityDecision2: \\nlawType: Infrequently litigated statutes\\nlawSupp: Infrequently litigated statutes\\nlawMinor: 18 U.S.C. € 398\\nmajOpinWriter: WODouglas \\tDouglas, William ( 04/17/1939 - 11/12/1975 )\\nmajOpinAssigner: FMVinson \\tVinson, Fred ( 06/24/1946 - 09/08/1953 )\\nsplitVote: first vote on issue/legal provision\\nmajVotes: 6\\nminVotes: 3'"
            ]
          },
          "execution_count": 14,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "cases[1]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "puwtnglTqb-l"
      },
      "outputs": [],
      "source": [
        "docs = text_splitter.split_documents(documents)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "3dV_xEaVDFt2"
      },
      "outputs": [],
      "source": [
        "from langchain_community.embeddings.sentence_transformer import (\n",
        "    SentenceTransformerEmbeddings,\n",
        ")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "6_gQyJmRzEYM"
      },
      "outputs": [],
      "source": [
        "embedding_function = SentenceTransformerEmbeddings(model_name=\"all-MiniLM-L6-v2\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "FXj1mhZaqxMe"
      },
      "outputs": [],
      "source": [
        "from langchain.vectorstores import Chroma"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ChMywMcGwt0g"
      },
      "outputs": [],
      "source": [
        "persist_directory = 'docs/chroma/'"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "7fZ1RUAaxiT1"
      },
      "outputs": [],
      "source": [
        "import os"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "nnCtyJvCqwDl"
      },
      "outputs": [],
      "source": [
        "db = Chroma.from_texts(cases, embedding_function)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "IjmgO_JL1vQk"
      },
      "outputs": [],
      "source": [
        "coll = db.get()\n",
        "\n",
        "db.delete_collection()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "QqvHUCRnqo07"
      },
      "outputs": [],
      "source": [
        "lm = dspy.GROQ(model='mixtral-8x7b-32768', api_key =\"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\",max_tokens=1000 )\n",
        "dspy.configure(lm=lm)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "H_5QM6j-Wz4_"
      },
      "outputs": [],
      "source": [
        "class RAG(dspy.Module):\n",
        "    def __init__(self, num_passages=3):\n",
        "        super().__init__()\n",
        "        self.retrieve = db\n",
        "        self.generate_answer = dspy.ChainOfThought(\"context, question -> answer\")\n",
        "\n",
        "    def forward(self, question):\n",
        "        context = self.retrieve.max_marginal_relevance_search(question,k=3)\n",
        "        answer = self.generate_answer(context=context, question=question)\n",
        "        return answer"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "W69MDzN1S9FD"
      },
      "outputs": [],
      "source": [
        "from dspy.teleprompt import BootstrapFewShot"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Ep3o-UAISi5Q"
      },
      "outputs": [],
      "source": [
        "uncompiled_rag = RAG()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "YPu5ibHHUc7i"
      },
      "outputs": [],
      "source": [
        "print(uncompiled_rag(\"Give me an example of case where chief was Warren\").answer)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "g4-1o94iVRGc"
      },
      "outputs": [],
      "source": [
        "lm.inspect_history(n=1)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "JjBD4R0mo61h"
      },
      "outputs": [],
      "source": [
        "import pandas as pd"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "jTJ45jkuo9Jk"
      },
      "outputs": [],
      "source": [
        "from google.colab import files\n",
        "files.upload()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "VEG7-PxHvt0J"
      },
      "outputs": [],
      "source": [
        "import numpy as np"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "bZT-ElyBpDTI"
      },
      "outputs": [],
      "source": [
        "df = pd.read_csv(\"output.csv\", encoding=\"cp1252\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "SxMzRwkqpYdK"
      },
      "outputs": [],
      "source": [
        "df.head()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "f1MwjnV-vhxt"
      },
      "outputs": [],
      "source": [
        "cases_from_df = np.array_split(df, len(df))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "LGczJksYwga-"
      },
      "outputs": [],
      "source": [
        "!pip install chromadb datasets"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "OXl5cHLawfCh"
      },
      "outputs": [],
      "source": [
        "from datasets import load_dataset"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 10,
      "metadata": {
        "id": "xGMAvoabw9oa"
      },
      "outputs": [],
      "source": [
        "client = chromadb.Client()\n",
        "collection = client.create_collection(\"Supreme_court_decisions\")\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 11,
      "metadata": {
        "id": "k8BVnccB3vlf"
      },
      "outputs": [],
      "source": [
        "client.delete_collection(\"Supreme_court_decisions\")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "xgFKKPqUy4jK"
      },
      "outputs": [],
      "source": [
        "cases_from_df[1]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Clqi7Myv5_HV"
      },
      "outputs": [],
      "source": [
        "from langchain.chains.query_constructor.base import AttributeInfo"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "kw2zkd0Y24EZ"
      },
      "outputs": [],
      "source": [
        "metadata_field_info = [\n",
        "    AttributeInfo(\n",
        "        name=\"caseId\",\n",
        "        description=\"This is the first of four unique internal identification numbers. The first four digits are the term. The next four are the case within the term (starting at 001 and counting up).\",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "    AttributeInfo(\n",
        "        name=\"docketId\",\n",
        "        description=\" This is the second of four unique internal identification numbers.The first four digits are the term. The next four are the case within the term (starting at 001 and counting up). The last two are the number of dockets consolidated under the U.S. Reports citation (starting at 01 and counting up).  \",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "    AttributeInfo(\n",
        "        name=\"caseIssuesId\",\n",
        "        description=\"This is the third of four unique internal identification numbers. The first four digits are the term. The next four are the case within the term (starting at 001 and counting up). The next two are the number of dockets consolidated under the U.S. Reports citation (starting at 01 and counting up). The last two are the number of issues and legal provisions within the case (starting at 01 and counting up). \",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "    AttributeInfo(\n",
        "        name=\"voteId\",\n",
        "        description=\"This is the fourth of four unique internal identification numbers. The first four digits are the term. The next four are the case within the term (starting at 001 and counting up). The next two are the number of dockets consolidated under the U.S. Reports citation (starting at 01 and counting up). The next two are the number of issues and legal provisions within the case (starting at 01 and counting up). The next two indicate a split vote within an issue or legal provision (01 for only one vote; 02 if a split vote). The final two represent the vote in the case (usually runs 01 to 09, but fewer if less than all justices participated). \",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "    AttributeInfo(\n",
        "        name=\"usCite\",\n",
        "        description=\" Provides the citation to each case from the official United States Reports (US) and the three major unofficial Reports\",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "    AttributeInfo(\n",
        "        name=\"sctCite\",\n",
        "        description=\"Provides the citation to each case from theSupreme Court Reporter\",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "    AttributeInfo(\n",
        "        name=\"ledCite\",\n",
        "        description=\"Provides the citation to each case from the Lawyers' Edition of the United States Reports(LEd)\",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "     AttributeInfo(\n",
        "        name=\"lexisCite\",\n",
        "        description=\"Provides the citation to each case from the LEXIS cite\",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "     AttributeInfo(\n",
        "        name=\"docket\",\n",
        "        description=\"This variable contains the docket number that the Supreme Court has assigned to the case. Prior to the first two terms of the Burger Court (1969-1970), different cases coming to the Court in different terms could have the same docket number. The Court eliminated the possibility of such duplication by including the last two digits of the appropriate term before the assigned docket number. Since the 1971 Term, the Court has also operated with a single docket. Cases filed pursuant to the Court's appellate jurisdiction have a two-digit number corresponding to the term in which they were filed, followed by a hyphen and a number varying from one to five digits. Cases invoking the Court's original jurisdiction have a number followed by the abbreviation, `Orig` \",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "     AttributeInfo(\n",
        "        name=\"dateRearg\",\n",
        "        description=\"On those infrequent occasions when the Court orders that a case be reargued, this variable specifies the date of such argumen\",\n",
        "        type=\"date\",\n",
        "    ),\n",
        "      AttributeInfo(\n",
        "        name=\"jurisdiction\",\n",
        "        description=\"The Court uses a variety of means whereby it undertakes to consider cases that it has been petitioned to review. These are listed below. The most important ones are the writ of certiorari, the writ of appeal, and for legacy cases the writ of error, appeal, and certification. \",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "      AttributeInfo(\n",
        "        name=\"adminAction\",\n",
        "        description=\"This variable pertains to administrative agency activity occurring prior to the onset of litigation. Note that the activity may involve an administrative official as well as that of an agency. The general rule for an entry in this variable is whether administrative action occurred in the context of the case. Note too that this variable identifies the specific federal agency. If the action occurred in a state agency, adminAction is coded as 117 (State Agency). See the variable adminActionState for the identity of the state. \",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "      AttributeInfo(\n",
        "        name=\"adminActionState\",\n",
        "        description=\"Administrative action may be either state or federal. If administrative action was taken by a state or a subdivision thereof, this variable identifies the state\",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "      AttributeInfo(\n",
        "        name=\"threeJudgeFdc\",\n",
        "        description=\"This variable will be checked if the case was heard by a three-judge federal district court (occasionally called “as specially constituted district court”). Beginning in the early 1900s, Congress required three-judge district courts to hear certain kinds of cases. More modern-day legislation has reduced the kinds of lawsuits that must be heard by such a court.\",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "      AttributeInfo(\n",
        "        name=\"caseOriginState\",\n",
        "        description=\"If the case originated in a state court, this variable identifies the state\",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "      AttributeInfo(\n",
        "        name=\"caseSourceState\",\n",
        "        description=\"If the source of the case (i.e., the court whose decision the Supreme Court reviewed) is a state court, this variable identifies the state\",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "     AttributeInfo(\n",
        "        name=\"lcDisagreement\",\n",
        "        description=\"An entry in this variable indicates that the Supreme Court's majority opinion mentioned that one or more of the members of the court whose decision the Supreme Court reviewed dissented. The presence of such disagreement is limited to a statement to this effect somewhere in the majority opinion. I.e, `divided,` `dissented,` `disagreed,` `split.` A reference, without more, to the `majority` or `plurality` does not necessarily evidence dissent. The other judges may have concurred. \",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "       AttributeInfo(\n",
        "        name=\"certReason\",\n",
        "        description=\"This variable provides the reason, if any, that the Court gives for granting the petition for certiorari. If the case did not arise on certiorari, this variable will be so coded even if the Court provides a reason why it agreed to hear the case. The Court, however, rarely provides a reason for taking jurisdiction by writs other than certiorari. \",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "        AttributeInfo(\n",
        "        name=\"lcDisposition\",\n",
        "        description=\"This variable specifies the treatment the court whose decision the Supreme Court reviewed accorded the decision of the court it reviewed; e.g., whether the court below the Supreme Court---typically a federal court of appeals or a state supreme court---affirmed, reversed, remanded, etc. the decision of the court it reviewed---typically a trial court. lcDisposition will not contain an entry if the decision the Supreme Court reviewed is that of a trial court or if the case arose under the Supreme Court's original jurisdiction (see the jurisdiction variable). The former occurs frequently in the legacy data. \",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "\n",
        "      AttributeInfo(\n",
        "        name=\"lcDispositionDirection\",\n",
        "        description=\"lcDispositionDirection permits determination of whether the Supreme Court's disposition of the case upheld or overturned a liberal or a conservative lower court decision. \",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "      AttributeInfo(\n",
        "        name=\"declarationUncon\",\n",
        "        description=\"An entry in this variable indicates that the Court either declared unconstitutional an act of Congress; a state or territorial statute, regulation, or constitutional provision; or a municipal or other local ordinance. In coding this variable we consulted several sources. Most helpful was the Congressional Research Service's Constitution of the United States of America: Analysis and Interpretation (CONAN) (https://www.congress.gov/constitution-annotated) and the appendix to volume 131 of the U.S. Reports. \",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "      AttributeInfo(\n",
        "        name=\"caseDisposition\",\n",
        "        description=\"\"\"The treatment the Supreme Court accorded the court whose decision it reviewed is contained in this variable; e.g., affirmed, vacated, reversed and remanded, etc. The values here are the same as those for lcDisposition (how the court whose decision the Supreme Court reviewed disposed of the case). For original jurisdiction cases, this variable will be empty unless the Court's disposition falls under 1 or 9 below (stay, petition, or motion granted; petition denied or appeal dismissed). For cases in which the Court granted a motion to dismiss, caseDisposition is coded as 9 (petition denied or appeal dismissed). There is \"no disposition\" if the Court denied a motion to dismiss. \"\"\",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "       AttributeInfo(\n",
        "        name=\"caseDispositionUnusual\",\n",
        "        description=\"An entry (1) will appear in this variable to signify that the Court made an unusual disposition of the cited case which does not match the coding scheme of the preceding variable. The disposition that appears closest to the unusual one made by the Court should be selected for inclusion in the preceding variable, caseDisposition. \",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "      AttributeInfo(\n",
        "        name=\"partyWinning\",\n",
        "        description=\"\"\"This variable indicates whether the petitioning party (i.e., the plaintiff or the appellant) emerged victorious. The victory the Supreme Court provided the petitioning party may not have been total and complete (e.g., by vacating and remanding the matter rather than an unequivocal reversal), but the disposition is nonetheless a favorable one.\n",
        "With some adjustments, we coded this variable according to the following rules:\n",
        "The petitioning party lost if the Supreme Court affirmed (caseDisposition=2) or dismissed the case/denied the petition (caseDisposition=9).\n",
        "The petitioning party won in part or in full if the Supreme Court reversed (caseDisposition=3), reversed and remanded (caseDisposition= 4), vacated and remanded (caseDisposition=5), affirmed and reversed in part (caseDisposition=6), affirmed and reverse in part and remanded (caseDisposition=7), or vacated (caseDisposition=8)\n",
        "The petitioning party won or lost may be unclear if the Court certified to/from a lower court. \"\"\",\n",
        "        type=\"integer\",\n",
        "    ),\n",
        "    AttributeInfo(\n",
        "        name=\"precedentAlteration\",\n",
        "        description=\"\"\"A \"1\" will appear in this variable if the majority opinion effectively says that the decision in this case \"overruled\" one or more of the Court's own precedents. Occasionally, in the absence of language in the prevailing opinion, the dissent will state clearly and persuasively that precedents have been formally altered: e.g., the two landmark reapportionment cases: Baker v. Carr, 369 U.S. 186 (1962), and Gray v. Sanders, 372 U.S. 368 (1963). Once in a great while the majority opinion will state--again in so many words--that an earlier decision overruled one of the Court's own precedents, even though that earlier decision nowhere says so. E.g, Patterson v. McLean Credit Union, 485 U.S. 617 (1988), in which the majority said that Braden v. 30th Judicial Circuit of Kentucky, 410 U.S. 484, 35 L Ed 2d 443 (1973) overruled a 1948 decision. On the basis of this later language, the earlier decision will contain a \"1\" in this variable. Alteration also extends to language in the majority opinion that states that a precedent of the Supreme Court has been \"disapproved,\" or is \"no longer good law.\" \"\"\",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "\n",
        "     AttributeInfo(\n",
        "        name=\"voteUnclear\",\n",
        "        description=\"\"\"The votes in a case are those specified in the opinions.\n",
        "Do note, however, that the majority opinion in a number of Marshall Court decisions reports that unnamed justices were in disagreement about the resolution of the case. These do not identify who the dissenters were. We, therefore, look to the majority opinion itself to specify who voted how. \"\"\",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "    AttributeInfo(\n",
        "        name=\"decisionDirectionDissent\",\n",
        "        description=\"\"\"Once in a great while the majority as well as the dissenting opinion in a case will both support or, conversely, oppose the issue to which the case pertains. For example, the majority and the dissent may both assert that the rights of a person accused of crime have been violated. The only difference between them is that the majority votes to reverse the accused's conviction and remand the case for a new trial, while the dissent holds that the accused's conviction should be reversed, period. In such cases, the entry in the decisionDirection variable should be determined relative to whether the majority or the dissent more substantially supported the issue to which the case pertains, and an entry should appear in this variable. In the foregoing example, the direction of decision variable (decisionDirection) should show a 0(conservative) because the majority provided the person accused of crime with less relief than does the dissent, and direction based on dissent should show a 1 (liberal) The person accused of crime actually won the case, but won less of a victory than the dissent would have provided. \"\"\",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "     AttributeInfo(\n",
        "        name=\"authorityDecision1\",\n",
        "        description=\"\"\"This variable and the next one (authorityDecision2) specify the bases on which the Supreme Court rested its decision with regard to each legal provision that the Court considered in the case (see variable lawType).\n",
        "\n",
        "Neither of them lends itself to objectivity. Many cases arguably rest on more than two bases for decision. Given\n",
        "that the Court's citation of its precedents also qualifies as a common law decision and that most every case can be considered as at least partially based thereon, common law is the default basis for the Court's decisions. With the exception of decrees and brief non-orally argued decisions you may safely add common law to those cases lacking a second basis for decision. \"\"\",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "     AttributeInfo(\n",
        "        name=\"authorityDecision2\",\n",
        "        description=\"See variable Authority for Decision 1 (authorityDecision1). \",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "      AttributeInfo(\n",
        "        name=\"lawType\",\n",
        "        description=\"This variable and its components identify the constitutional provision(s), statute(s), or court rule(s) that the Court considered in the case\",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "    AttributeInfo(\n",
        "        name=\"lawSupp\",\n",
        "        description=\" The difference between them is that lawSupp and lawMinor are coded finely; they identify the specific law, constitutional provision or rule at issue (e.g., Article I, Section 1; the Federal Election Campaign Act; the Federal Rules of Evidence). lawType is coded more broadly (e.g., constitution, federal statute, court rules).\",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "    AttributeInfo(\n",
        "        name=\"lawMinor\",\n",
        "        description=\"This variable, lawMinor, is reserved for infrequently litigated statutes. Statutes substantially absent from the decision making of the modern Courts will be found in this variable. For these, lawMinor identifies the law at issue. Note: This is a string variable. \",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "    AttributeInfo(\n",
        "        name=\"majOpinWriter\",\n",
        "        description=\"This variable identifies the author of the Court's opinion or judgment, as the case may be.\",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "    AttributeInfo(\n",
        "        name=\"majOpinAssigner\",\n",
        "        description=\"This variable identifies the assigner of the opinion or judgment of the Court, as the case may be. These data are drawn from the membership in the final (report vote) coalition and from the rules governing opinion assignment: If the chief justice is a member of the majority vote coalition at the conference vote, he assigns the opinion; if not, the senior associate justice who is a member of the majority at the conference vote does so. According to several scholarly studies, considerable voting shifts occur between the final conference vote (where the assignment is made) and the vote that appears in the Reports. As a result, in approximately 16 percent of the cases, a person other than the one identified by the database actually assigned the opinion. \",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "    AttributeInfo(\n",
        "        name=\"splitVote\",\n",
        "        description=\"This variable indicates whether the vote variables (e.g., majVotes, minVotes) pertain to the vote on the first or second issue (or legal provision). Because split votes are so rare over 99 percent of the votes are on the first issue. \",\n",
        "        type=\"string\",\n",
        "    ),\n",
        "     AttributeInfo(\n",
        "        name=\"majVotes\",\n",
        "        description=\"This variable specifies the number of justices voting in the majority; minVotes indicates the number of justices voting in dissent. \",\n",
        "        type=\"integer\",\n",
        "    ),\n",
        "     AttributeInfo(\n",
        "        name=\"minVotes\",\n",
        "        description=\"This variable specifies the number of votes in dissent. Only dissents on the merits are specified in this variable.Justices who dissent from a denial or dismissal of certiorari or who disagree with the Court's assertion of jurisdiction count as not participating in the decision. \",\n",
        "        type=\"integer\",\n",
        "    ),\n",
        "]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "Vf5p_bGT61IB"
      },
      "outputs": [],
      "source": [
        "important_columns = [\"decisionType\", \"dateDecision\",\"term\",\"naturalCourt\", \"caseName\",\"chief\", \"dateArgument\",\"petitioner\", \"petitionerState\",\"respondent\",\"respondentState\",\"caseOrigin\",\"caseSource\", \"issue\",\"issueArea\",\"decisionDirection\"]\n",
        "metadata_columns = [item for item in df.columns.tolist() if item not in important_columns]"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "5J11dAIPxFD6"
      },
      "outputs": [],
      "source": [
        "for i in range(len(cases_from_df)):\n",
        "  res=\"\"\n",
        "  metadata=\"\"\n",
        "  for column_name in important_columns:\n",
        "    res+=column_name+\": \" + str(cases_from_df[i][column_name].item()) +\"\\n\"\n",
        "  for column_name in metadata_columns:\n",
        "    metadata+=column_name+\": \" + str(cases_from_df[i][column_name].item()) + \"\\n\"\n",
        "  collection.add(\n",
        "      ids=[str(i)],\n",
        "      documents=res,\n",
        "       metadatas = [{\"documents\": metadata}])\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "6JLbvu4T_JL8"
      },
      "outputs": [],
      "source": [
        "langchain_chroma = Chroma(\n",
        "    client=client,\n",
        "    collection_name=\"Supreme_court_decisions\",\n",
        "    embedding_function=embedding_function\n",
        ")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "IMdUMIj5Fht2"
      },
      "outputs": [],
      "source": [
        "from langchain.chains.query_constructor.base import AttributeInfo\n",
        "from langchain.retrievers.self_query.base import SelfQueryRetriever"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "-e1lgORZG7o-"
      },
      "outputs": [],
      "source": [
        "document_content_description = \"Case details\""
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "KmFZseqEYRul"
      },
      "outputs": [],
      "source": [
        "!pip install langchain-openai"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "4ri0D9NmjKjd"
      },
      "outputs": [],
      "source": [
        "from langchain.chains.query_constructor.base import (\n",
        "    StructuredQueryOutputParser,\n",
        "    get_query_constructor_prompt,\n",
        "    AttributeInfo\n",
        ")"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "8NsZCmjljeH0"
      },
      "outputs": [],
      "source": [
        "question = \"Can you give me a details about case where chief was Warren.\""
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "vsFsqCmS9ZTy"
      },
      "outputs": [],
      "source": [
        "class RAG2(dspy.Module):\n",
        "    def __init__(self):\n",
        "        super().__init__()\n",
        "        self.retrieve = langchain_chroma\n",
        "        self.generate_answer = dspy.ChainOfThought(\"context, question -> answer\")\n",
        "\n",
        "    def forward(self, question):\n",
        "        context = self.retrieve.max_marginal_relevance_search(question,k=1)\n",
        "        print(context)\n",
        "        answer = self.generate_answer(context=context, question=question)\n",
        "        return dspy.Prediction(answer=answer.answer)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "iB30BVP49utz"
      },
      "outputs": [],
      "source": [
        "rag2 = RAG2()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "LIYTeLfS90CP",
        "outputId": "c855c6a9-563a-477f-96ad-d7c2032fe636"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "The case name is UNITED STATES v. BRAMBLETT, and it was decided on 4/4/1955. The case was argued on 2/7/1955, and it originated from the District Of Columbia U.S. District Court. The decision direction of the case was conservative, and Earl Warren was the chief justice at the time.\n"
          ]
        }
      ],
      "source": [
        "print(rag2(\"Can you give me a details about case where chief was Warren.\").answer)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1jR5st_6_OD9",
        "outputId": "3eafce31-5bcc-4011-cdce-3befcd893bbc"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "This case originated in the State Trial Court.\n"
          ]
        }
      ],
      "source": [
        "print(rag2(\"In which court this case originated\").answer)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "4fQXAD_1_TtV",
        "outputId": "d6c68c24-4bb5-4e24-8b6d-b31cabb44624"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "The case KAWAKITA v. UNITED STATES was decided on 6/2/1952.\n"
          ]
        }
      ],
      "source": [
        "print(rag2(\"When this case happened\").answer)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/",
          "height": 627
        },
        "id": "kwY3HYVoKHZr",
        "outputId": "50eb6fa4-9296-4082-ab87-3a4aad9a3274"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\n",
            "\n",
            "\n",
            "Given the fields `context`, `question`, produce the fields `answer`.\n",
            "\n",
            "---\n",
            "\n",
            "Follow the following format.\n",
            "\n",
            "Context: ${context}\n",
            "\n",
            "Question: ${question}\n",
            "\n",
            "Reasoning: Let's think step by step in order to ${produce the answer}. We ...\n",
            "\n",
            "Answer: ${answer}\n",
            "\n",
            "---\n",
            "\n",
            "Context: «page_content='decisionType: opinion of the court (orally argued)\\ndateDecision: 4/4/1955\\nterm: 1954\\nnaturalCourt: Warren 3 \\tMarch 28, 1955 - October 15, 1956\\ncaseName: UNITED STATES v. BRAMBLETT\\nchief: Warren\\ndateArgument: 2/7/1955\\npetitioner: United States\\npetitionerState: nan\\nrespondent: person accused, indicted, or suspected of crime\\nrespondentState: nan\\ncaseOrigin: District Of Columbia U.S. District Court\\ncaseSource: District Of Columbia U.S. District Court\\nissue: statutory construction of criminal laws: false statements (cf. statutory construction of criminal laws: perjury)\\nissueArea: Criminal Procedure\\ndecisionDirection: conservative\\n' metadata={'adminAction': 'nan', 'adminActionState': 'nan', 'authorityDecision1': 'statutory construction', 'authorityDecision2': 'nan', 'caseDisposition': 'reversed', 'caseDispositionUnusual': 'no unusual disposition specified', 'caseId': '1954-048', 'caseIssuesId': '1954-048-01-01', 'caseOriginState': 'nan', 'caseSourceState': 'nan', 'certReason': 'case did not arise on cert or cert not granted', 'dateRearg': 'nan', 'decisionDirectionDissent': 'dissent in opposite direction', 'declarationUncon': 'no declaration of unconstitutionality', 'docket': '159', 'docketId': '1954-048-01', 'jurisdiction': 'appeal', 'lawMinor': '18 U.S.C.  1001', 'lawSupp': 'Infrequently litigated statutes', 'lawType': 'Infrequently litigated statutes', 'lcDisagreement': 'no mention that dissent occurred', 'lcDisposition': 'nan', 'lcDispositionDirection': 'liberal', 'ledCite': '99 L. Ed. 2d 594', 'lexisCite': '1955 U.S. LEXIS 975', 'majOpinAssigner': 'HLBlack \\tBlack, Hugo ( 08/19/1937 - 09/17/1971 )', 'majOpinWriter': 'SFReed \\tReed, Stanley ( 01/31/1938 - 02/25/1957 )', 'majVotes': '6', 'minVotes': '0', 'partyWinning': 'petitioning party received a favorable disposition', 'precedentAlteration': 'no determinable alteration of precedent', 'sctCite': '75 S. Ct. 504', 'splitVote': 'first vote on issue/legal provision', 'threeJudgeFdc': 'no mention that a 3-judge ct heard case', 'usCite': '348 U.S. 503', 'voteId': '1954-048-01-01-01', 'voteUnclear': 'vote clearly specified'}»\n",
            "\n",
            "Question: Can you give me a details about case where chief was Warren.\n",
            "\n",
            "Reasoning: Let's think step by step in order to\u001b[32m give you details about a case where the chief was Warren. According to the context provided, the case name is \"UNITED STATES v. BRAMBLETT\". The context also indicates that the chief was Warren, and the case was decided on April 4, 1955. The case involved the statutory construction of criminal laws related to false statements. The decision direction of the case was conservative, and it reversed the decision of the District of Columbia U.S. District Court. The case originated from the District of Columbia U.S. District Court.\n",
            "\n",
            "Answer: The case where the chief was Warren is \"UNITED STATES v. BRAMBLETT\", decided on April 4, 1955. The case involved the statutory construction of criminal laws related to false statements, and it reversed the decision of the District of Columbia U.S. District Court.\u001b[0m\u001b[31m \t (and 801 other completions)\u001b[0m\n",
            "\n",
            "\n",
            "\n"
          ]
        },
        {
          "data": {
            "application/vnd.google.colaboratory.intrinsic+json": {
              "type": "string"
            },
            "text/plain": [
              "'\\n\\n\\nGiven the fields `context`, `question`, produce the fields `answer`.\\n\\n---\\n\\nFollow the following format.\\n\\nContext: ${context}\\n\\nQuestion: ${question}\\n\\nReasoning: Let\\'s think step by step in order to ${produce the answer}. We ...\\n\\nAnswer: ${answer}\\n\\n---\\n\\nContext: «page_content=\\'decisionType: opinion of the court (orally argued)\\\\ndateDecision: 4/4/1955\\\\nterm: 1954\\\\nnaturalCourt: Warren 3 \\\\tMarch 28, 1955 - October 15, 1956\\\\ncaseName: UNITED STATES v. BRAMBLETT\\\\nchief: Warren\\\\ndateArgument: 2/7/1955\\\\npetitioner: United States\\\\npetitionerState: nan\\\\nrespondent: person accused, indicted, or suspected of crime\\\\nrespondentState: nan\\\\ncaseOrigin: District Of Columbia U.S. District Court\\\\ncaseSource: District Of Columbia U.S. District Court\\\\nissue: statutory construction of criminal laws: false statements (cf. statutory construction of criminal laws: perjury)\\\\nissueArea: Criminal Procedure\\\\ndecisionDirection: conservative\\\\n\\' metadata={\\'adminAction\\': \\'nan\\', \\'adminActionState\\': \\'nan\\', \\'authorityDecision1\\': \\'statutory construction\\', \\'authorityDecision2\\': \\'nan\\', \\'caseDisposition\\': \\'reversed\\', \\'caseDispositionUnusual\\': \\'no unusual disposition specified\\', \\'caseId\\': \\'1954-048\\', \\'caseIssuesId\\': \\'1954-048-01-01\\', \\'caseOriginState\\': \\'nan\\', \\'caseSourceState\\': \\'nan\\', \\'certReason\\': \\'case did not arise on cert or cert not granted\\', \\'dateRearg\\': \\'nan\\', \\'decisionDirectionDissent\\': \\'dissent in opposite direction\\', \\'declarationUncon\\': \\'no declaration of unconstitutionality\\', \\'docket\\': \\'159\\', \\'docketId\\': \\'1954-048-01\\', \\'jurisdiction\\': \\'appeal\\', \\'lawMinor\\': \\'18 U.S.C.  1001\\', \\'lawSupp\\': \\'Infrequently litigated statutes\\', \\'lawType\\': \\'Infrequently litigated statutes\\', \\'lcDisagreement\\': \\'no mention that dissent occurred\\', \\'lcDisposition\\': \\'nan\\', \\'lcDispositionDirection\\': \\'liberal\\', \\'ledCite\\': \\'99 L. Ed. 2d 594\\', \\'lexisCite\\': \\'1955 U.S. LEXIS 975\\', \\'majOpinAssigner\\': \\'HLBlack \\\\tBlack, Hugo ( 08/19/1937 - 09/17/1971 )\\', \\'majOpinWriter\\': \\'SFReed \\\\tReed, Stanley ( 01/31/1938 - 02/25/1957 )\\', \\'majVotes\\': \\'6\\', \\'minVotes\\': \\'0\\', \\'partyWinning\\': \\'petitioning party received a favorable disposition\\', \\'precedentAlteration\\': \\'no determinable alteration of precedent\\', \\'sctCite\\': \\'75 S. Ct. 504\\', \\'splitVote\\': \\'first vote on issue/legal provision\\', \\'threeJudgeFdc\\': \\'no mention that a 3-judge ct heard case\\', \\'usCite\\': \\'348 U.S. 503\\', \\'voteId\\': \\'1954-048-01-01-01\\', \\'voteUnclear\\': \\'vote clearly specified\\'}»\\n\\nQuestion: Can you give me a details about case where chief was Warren.\\n\\nReasoning: Let\\'s think step by step in order to\\x1b[32m give you details about a case where the chief was Warren. According to the context provided, the case name is \"UNITED STATES v. BRAMBLETT\". The context also indicates that the chief was Warren, and the case was decided on April 4, 1955. The case involved the statutory construction of criminal laws related to false statements. The decision direction of the case was conservative, and it reversed the decision of the District of Columbia U.S. District Court. The case originated from the District of Columbia U.S. District Court.\\n\\nAnswer: The case where the chief was Warren is \"UNITED STATES v. BRAMBLETT\", decided on April 4, 1955. The case involved the statutory construction of criminal laws related to false statements, and it reversed the decision of the District of Columbia U.S. District Court.\\x1b[0m\\x1b[31m \\t (and 801 other completions)\\x1b[0m\\n\\n\\n'"
            ]
          },
          "execution_count": 167,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "lm.inspect_history(n=1)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "_4QwzW2sqnym"
      },
      "outputs": [],
      "source": [
        "from langchain.chains import create_history_aware_retriever\n",
        "from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "bp3GMHULsDPs"
      },
      "outputs": [],
      "source": [
        "from langchain.chains import create_retrieval_chain\n",
        "from langchain.chains.combine_documents import create_stuff_documents_chain"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "jxohKqBM7Xol"
      },
      "outputs": [],
      "source": [
        "class RagWithMemory():\n",
        "  def __init__(self):\n",
        "    self.rag = RAG2()\n",
        "    self.chat_history = []\n",
        "  def forward(self, question):\n",
        "    new_prompt_tempalte = f\"Consider previous chat history:{self.chat_history} \\nConsider this information in your following answers\\n Question: {question}\"\n",
        "    pred = self.rag(new_prompt_tempalte)\n",
        "    answer = pred.answer\n",
        "\n",
        "    self.chat_history.append(f\"Question: {question} \\nAnswer: {answer}\")\n",
        "    return answer"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "kDZ6moPOscUv"
      },
      "outputs": [],
      "source": [
        "rag_with_memory = RagWithMemory()"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "5paX5TwT99u0",
        "outputId": "acbca99c-327e-4c4a-8741-e4171a12babc"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "The case \"BROWNELL, ATTORNEY GENERAL, SUCCESSOR TO THE ALIEN PROPERTY CUSTODIAN, v. SINGER\" is an example of a case where the chief was Warren.\n"
          ]
        }
      ],
      "source": [
        "print(rag_with_memory.forward(\"Give me example of a case where chief was Warren.\"))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "7RvGOzgK_ErQ",
        "outputId": "2f08e3ad-6749-4048-b285-cd21e581fe80"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "The case \"BROWNELL, ATTORNEY GENERAL, SUCCESSOR TO THE ALIEN PROPERTY CUSTODIAN, v. SINGER\" was decided on April 5, 1954, during the 1953 term of court. The chief justice was Warren. The case originated from the State Supreme Court, and the issue was the priority of federal fiscal claims over those of states or private entities. The decision direction was liberal, and the case was reversed, with the petitioning party receiving a favorable disposition. The vote was split, with 5 votes in favor and 3 votes against. The case did not result in a declaration of unconstitutionality and involved federal common law.\n"
          ]
        }
      ],
      "source": [
        "print(rag_with_memory.forward(\"Give me details about this case\"))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "Ha7eu850_Y38",
        "outputId": "55813d5e-c7f8-4ae2-e278-0b559b3b2719"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "The case was decided on April 5, 1954.\n"
          ]
        }
      ],
      "source": [
        "print(rag_with_memory.forward(\"When this has happened\"))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "RuVhzo0PA4uT",
        "outputId": "6673ec32-10de-404f-b293-d99c55842333"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "The issue area of the case \"BROWNELL, ATTORNEY GENERAL, SUCCESSOR TO THE ALIEN PROPERTY CUSTODIAN, v. SINGER\" was Federal Taxation.\n"
          ]
        }
      ],
      "source": [
        "print(rag_with_memory.forward(\"What was the issue area of this case\"))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "QPuC-akIBAAI",
        "outputId": "60e2e31c-27c9-46de-9571-b0be769969a8"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m8.5/8.5 MB\u001b[0m \u001b[31m15.4 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m207.3/207.3 kB\u001b[0m \u001b[31m16.8 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m6.9/6.9 MB\u001b[0m \u001b[31m33.3 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m83.0/83.0 kB\u001b[0m \u001b[31m8.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[2K     \u001b[90m━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━\u001b[0m \u001b[32m62.7/62.7 kB\u001b[0m \u001b[31m6.0 MB/s\u001b[0m eta \u001b[36m0:00:00\u001b[0m\n",
            "\u001b[?25h"
          ]
        }
      ],
      "source": [
        "!pip install -q streamlit"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 12,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "sqe_wEJsJUN2",
        "outputId": "8319b7d7-ba04-44f9-902b-1e59a4ae4c53"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "Overwriting app.py\n"
          ]
        }
      ],
      "source": [
        "%%writefile app.py\n",
        "import dspy\n",
        "import groq\n",
        "import colbert\n",
        "import chromadb\n",
        "from sentence_transformers import SentenceTransformer\n",
        "from langchain.text_splitter import TextSplitter\n",
        "import streamlit as st\n",
        "import pandas as pd\n",
        "from langchain.vectorstores import Chroma\n",
        "import numpy as np\n",
        "from langchain_community.embeddings.sentence_transformer import (\n",
        "    SentenceTransformerEmbeddings,\n",
        ")\n",
        "import streamlit as st\n",
        "class CaseSplitter(TextSplitter):\n",
        "    def __init__(self):\n",
        "        super().__init__()\n",
        "\n",
        "    def split_text(self, file):\n",
        "        cases = file.strip().split(\"\\n\\nRow \")\n",
        "        return cases\n",
        "\n",
        "\n",
        "lm = dspy.GROQ(model='mixtral-8x7b-32768', api_key=\"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\",\n",
        "               max_tokens=1000)\n",
        "dspy.configure(lm=lm)\n",
        "df = pd.read_csv(\"output.csv\", encoding=\"cp1252\")\n",
        "\n",
        "\n",
        "def create_collection(client):\n",
        "    cases_from_df = np.array_split(df, len(df))\n",
        "    collection = client.get_or_create_collection(\"Supreme_court_decisions\")\n",
        "    important_columns = [\"decisionType\", \"dateDecision\", \"term\", \"naturalCourt\", \"caseName\", \"chief\", \"dateArgument\",\n",
        "                         \"petitioner\", \"petitionerState\", \"respondent\", \"respondentState\", \"caseOrigin\", \"caseSource\",\n",
        "                         \"issue\", \"issueArea\", \"decisionDirection\"]\n",
        "    metadata_columns = [item for item in df.columns.tolist() if item not in important_columns]\n",
        "    for i in range(len(cases_from_df)):\n",
        "\n",
        "        res = \"\"\n",
        "        metadata = \"\"\n",
        "        for column_name in important_columns:\n",
        "            res += column_name + \": \" + str(cases_from_df[i][column_name].item()) + \"\\n\"\n",
        "        for column_name in metadata_columns:\n",
        "            metadata += column_name + \": \" + str(cases_from_df[i][column_name].item()) + \"\\n\"\n",
        "        collection.add(\n",
        "            ids=[str(i)],\n",
        "            documents=res,\n",
        "            metadatas=[{\"documents\": metadata}])\n",
        "\n",
        "\n",
        "if 'button_clicked' not in st.session_state:\n",
        "    st.session_state['button_clicked'] = False\n",
        "\n",
        "\n",
        "class RAG2(dspy.Module):\n",
        "    def __init__(self):\n",
        "        super().__init__()\n",
        "        client = chromadb.Client()\n",
        "        print(client.list_collections())\n",
        "        if \"Supreme_court_decisions\" not in [c.name for c in client.list_collections()]:\n",
        "            create_collection(client)\n",
        "        embedding_function = SentenceTransformerEmbeddings(model_name=\"all-MiniLM-L6-v2\")\n",
        "        langchain_chroma = Chroma(\n",
        "            client=client,\n",
        "            collection_name=\"Supreme_court_decisions\",\n",
        "            embedding_function=embedding_function\n",
        "        )\n",
        "        self.retrieve = langchain_chroma\n",
        "        self.generate_answer = dspy.ChainOfThought(\"context, question -> answer\")\n",
        "    def forward(self, question):\n",
        "        context = self.retrieve.max_marginal_relevance_search(question, k=1)\n",
        "        answer = self.generate_answer(context=context, question=question)\n",
        "        print(context)\n",
        "        return dspy.Prediction(answer=answer.answer)\n",
        "\n",
        "\n",
        "class RagWithMemory():\n",
        "    def __init__(self):\n",
        "        print(\"RagWithMemory\")\n",
        "        self.rag = RAG2()\n",
        "\n",
        "\n",
        "    def forward(self, question, history):\n",
        "        new_prompt_tempalte = f\"You are an AI assistant, which gives details about already existing Supreme Court decisions. Consider previous chat history:{history} \\nConsider this information in your following answers\\n Question: {question}\"\n",
        "\n",
        "        pred = self.rag(new_prompt_tempalte)\n",
        "        answer = pred.answer\n",
        "\n",
        "        return answer\n",
        "\n",
        "\n",
        "def get_llm_response(question, rag):\n",
        "    answer = rag.forward(question)\n",
        "    return answer\n",
        "\n",
        "\n",
        "st.title(\"💬 Chatbot\")\n",
        "st.caption(\"🚀 A streamlit chatbot powered by OpenAI LLM\")\n",
        "if \"messages\" not in st.session_state:\n",
        "    st.session_state[\"messages\"] = [{\"role\": \"assistant\", \"content\": \"How can I help you?\"}]\n",
        "\n",
        "for msg in st.session_state.messages:\n",
        "    st.chat_message(msg[\"role\"]).write(msg[\"content\"])\n",
        "\n",
        "if prompt := st.chat_input():\n",
        "    rag_with_memory = RagWithMemory()\n",
        "\n",
        "    st.chat_message(\"user\").write(prompt)\n",
        "    msg = rag_with_memory.forward(prompt, st.session_state.messages)\n",
        "    st.session_state.messages.append({\"role\": \"user\", \"content\": prompt})\n",
        "    st.session_state.messages.append({\"role\": \"assistant\", \"content\": msg})\n",
        "    st.chat_message(\"assistant\").write(msg)\n"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "fEjtkhRvMmLc"
      },
      "outputs": [],
      "source": [
        "!pip install -U langchain-community"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 13,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "1dZKWaj5Cjpd",
        "outputId": "a6ee46ee-d778-414b-f82b-39f270749187"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "34.148.101.162\n"
          ]
        }
      ],
      "source": [
        "!wget -q -O - ipv4.icanhazip.com"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": 15,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "_ov7TOBzDTdS",
        "outputId": "97eb7bde-53e6-4000-fd81-1864aa21a72f"
      },
      "outputs": [
        {
          "output_type": "stream",
          "name": "stdout",
          "text": [
            "\n",
            "Collecting usage statistics. To deactivate, set browser.gatherUsageStats to false.\n",
            "\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m\u001b[1m  You can now view your Streamlit app in your browser.\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[34m  Network URL: \u001b[0m\u001b[1mhttp://172.28.0.12:8501\u001b[0m\n",
            "\u001b[34m  External URL: \u001b[0m\u001b[1mhttp://34.148.101.162:8501\u001b[0m\n",
            "\u001b[0m\n",
            "\u001b[K\u001b[?25hnpx: installed 22 in 2.904s\n",
            "your url is: https://orange-cobras-remain.loca.lt\n",
            "/usr/local/lib/python3.10/dist-packages/langchain/vectorstores/__init__.py:35: LangChainDeprecationWarning: Importing vector stores from langchain is deprecated. Importing from langchain will no longer be supported as of langchain==0.2.0. Please import from langchain-community instead:\n",
            "\n",
            "`from langchain_community.vectorstores import Chroma`.\n",
            "\n",
            "To install langchain-community run `pip install -U langchain-community`.\n",
            "  warnings.warn(\n",
            "/usr/local/lib/python3.10/dist-packages/langchain/vectorstores/__init__.py:35: LangChainDeprecationWarning: Importing vector stores from langchain is deprecated. Importing from langchain will no longer be supported as of langchain==0.2.0. Please import from langchain-community instead:\n",
            "\n",
            "`from langchain_community.vectorstores import Chroma`.\n",
            "\n",
            "To install langchain-community run `pip install -U langchain-community`.\n",
            "  warnings.warn(\n",
            "RagWithMemory\n",
            "[]\n",
            "2024-05-07 09:58:50.698228: E external/local_xla/xla/stream_executor/cuda/cuda_dnn.cc:9261] Unable to register cuDNN factory: Attempting to register factory for plugin cuDNN when one has already been registered\n",
            "2024-05-07 09:58:50.698306: E external/local_xla/xla/stream_executor/cuda/cuda_fft.cc:607] Unable to register cuFFT factory: Attempting to register factory for plugin cuFFT when one has already been registered\n",
            "2024-05-07 09:58:50.700080: E external/local_xla/xla/stream_executor/cuda/cuda_blas.cc:1515] Unable to register cuBLAS factory: Attempting to register factory for plugin cuBLAS when one has already been registered\n",
            "2024-05-07 09:58:51.788324: W tensorflow/compiler/tf2tensorrt/utils/py_utils.cc:38] TF-TRT Warning: Could not find TensorRT\n",
            "[Document(page_content='decisionType: opinion of the court (orally argued)\\ndateDecision: 3/8/1954\\nterm: 1953\\nnaturalCourt: Warren 1 \\tOctober 05, 1953 - October 08, 1954\\ncaseName: ADAMS v. MARYLAND\\nchief: Warren\\ndateArgument: 1/7/1954\\npetitioner: witness, or person under subpoena\\npetitionerState: nan\\nrespondent: State\\nrespondentState: Maryland\\ncaseOrigin: State Trial Court\\ncaseSource: State Supreme Court\\nissue: self-incrimination, immunity from prosecution\\nissueArea: Criminal Procedure\\ndecisionDirection: liberal\\n', metadata={'documents': 'caseId: 1953-045\\ndocketId: 1953-045-01\\ncaseIssuesId: 1953-045-01-01\\nvoteId: 1953-045-01-01-01\\nusCite: 347 U.S. 179\\nsctCite: 74 S. Ct. 442\\nledCite: 98 L. Ed. 2d 608\\nlexisCite: 1954 U.S. LEXIS 2370\\ndocket: 271\\ndateRearg: nan\\njurisdiction: cert\\nadminAction: nan\\nadminActionState: nan\\nthreeJudgeFdc: no mention that a 3-judge ct heard case\\ncaseOriginState: 25.0\\ncaseSourceState: Maryland\\nlcDisagreement: no mention that dissent occurred\\ncertReason: no reason given\\nlcDisposition: affirmed\\nlcDispositionDirection: conservative\\ndeclarationUncon: no declaration of unconstitutionality\\ncaseDisposition: reversed and remanded\\ncaseDispositionUnusual: no unusual disposition specified\\npartyWinning: petitioning party received a favorable disposition\\nprecedentAlteration: no determinable alteration of precedent\\nvoteUnclear: vote clearly specified\\ndecisionDirectionDissent: dissent in opposite direction\\nauthorityDecision1: statutory construction\\nauthorityDecision2: nan\\nlawType: Infrequently litigated statutes\\nlawSupp: Infrequently litigated statutes\\nlawMinor: 18 U.S.C. € 3486\\nmajOpinWriter: HLBlack \\tBlack, Hugo ( 08/19/1937 - 09/17/1971 )\\nmajOpinAssigner: EWarren \\tWarren, Earl ( 10/05/1953 - 06/23/1969 )\\nsplitVote: first vote on issue/legal provision\\nmajVotes: 9\\nminVotes: 0\\n'})]\n",
            "/usr/local/lib/python3.10/dist-packages/langchain/vectorstores/__init__.py:35: LangChainDeprecationWarning: Importing vector stores from langchain is deprecated. Importing from langchain will no longer be supported as of langchain==0.2.0. Please import from langchain-community instead:\n",
            "\n",
            "`from langchain_community.vectorstores import Chroma`.\n",
            "\n",
            "To install langchain-community run `pip install -U langchain-community`.\n",
            "  warnings.warn(\n",
            "RagWithMemory\n",
            "[Collection(name=Supreme_court_decisions)]\n",
            "[Document(page_content='decisionType: opinion of the court (orally argued)\\ndateDecision: 3/8/1954\\nterm: 1953\\nnaturalCourt: Warren 1 \\tOctober 05, 1953 - October 08, 1954\\ncaseName: ADAMS v. MARYLAND\\nchief: Warren\\ndateArgument: 1/7/1954\\npetitioner: witness, or person under subpoena\\npetitionerState: nan\\nrespondent: State\\nrespondentState: Maryland\\ncaseOrigin: State Trial Court\\ncaseSource: State Supreme Court\\nissue: self-incrimination, immunity from prosecution\\nissueArea: Criminal Procedure\\ndecisionDirection: liberal\\n', metadata={'documents': 'caseId: 1953-045\\ndocketId: 1953-045-01\\ncaseIssuesId: 1953-045-01-01\\nvoteId: 1953-045-01-01-01\\nusCite: 347 U.S. 179\\nsctCite: 74 S. Ct. 442\\nledCite: 98 L. Ed. 2d 608\\nlexisCite: 1954 U.S. LEXIS 2370\\ndocket: 271\\ndateRearg: nan\\njurisdiction: cert\\nadminAction: nan\\nadminActionState: nan\\nthreeJudgeFdc: no mention that a 3-judge ct heard case\\ncaseOriginState: 25.0\\ncaseSourceState: Maryland\\nlcDisagreement: no mention that dissent occurred\\ncertReason: no reason given\\nlcDisposition: affirmed\\nlcDispositionDirection: conservative\\ndeclarationUncon: no declaration of unconstitutionality\\ncaseDisposition: reversed and remanded\\ncaseDispositionUnusual: no unusual disposition specified\\npartyWinning: petitioning party received a favorable disposition\\nprecedentAlteration: no determinable alteration of precedent\\nvoteUnclear: vote clearly specified\\ndecisionDirectionDissent: dissent in opposite direction\\nauthorityDecision1: statutory construction\\nauthorityDecision2: nan\\nlawType: Infrequently litigated statutes\\nlawSupp: Infrequently litigated statutes\\nlawMinor: 18 U.S.C. € 3486\\nmajOpinWriter: HLBlack \\tBlack, Hugo ( 08/19/1937 - 09/17/1971 )\\nmajOpinAssigner: EWarren \\tWarren, Earl ( 10/05/1953 - 06/23/1969 )\\nsplitVote: first vote on issue/legal provision\\nmajVotes: 9\\nminVotes: 0\\n'})]\n",
            "\u001b[34m  Stopping...\u001b[0m\n",
            "^C\n"
          ]
        }
      ],
      "source": [
        "! streamlit run app.py & npx localtunnel --port 8501"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyPyiBYyH1+JJ+IZW5fx6AxH",
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}