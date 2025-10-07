# Azure model name
GPT_4O = "gpt-4o"
GPT_4 = "gpt-4"
GPT_4O_MINI = "gpt-4o-mini"
GPT_4_TURBO = "gpt-4-turbo"
GPT_35_TURBO = "gpt-35-turbo"
GPT_4_1_MINI = "gpt-4.1-mini" # 1.6
GPT_4_1_NANO = "gpt-4.1-nano"
O1 = "o1"
O1_MINI = "o1-mini"
O4_MINI = "o4-mini"
GPT_4_1 = "gpt-4.1" 

OPENAI = "openai"
TOGETHER = "together"
AZURE = "azure"

INPUT_TEXT_TOKEN = "<INPUT_TEXT>"

DEEPSEEK_V3_0324 =  "DeepSeek-V3-0324"

GROK_3_MINI = "grok-3-mini"

PHI_4 = "Phi-4"

NAIVE_BAYES = "Naive Bayes"
SVM = "SVM"
ROBERTA_BASE = "roberta-base"

PROVIDER_MAPPING = {GPT_4O_MINI:OPENAI, 
                    GPT_4_1:OPENAI,
                    GPT_4_1_MINI:OPENAI,
                    GPT_4O: OPENAI,
                    GROK_3_MINI:AZURE,
                    PHI_4: AZURE,
                    DEEPSEEK_V3_0324: AZURE}

# Together model name
LLAMA_3_3_70B_INSTRUCT_TURBO_FREE = "meta-llama/Llama-3.3-70B-Instruct-Turbo-Free"

META_LLAMA_3_3_70B_INSTRUCT_TURBO = "meta-llama/Llama-3.3-70B-Instruct-Turbo" #$0.88

DEEPSEEK_R1_DISTILL_LLAMA_70B_FREE = "deepseek-ai/DeepSeek-R1-Distill-Llama-70B-free"

# DEEPSEEK_V3_0324 = "deepseek-ai/DeepSeek-V3" # $1.25

QWEN2_5_72B_INSTRUCT_TURBO = "Qwen/Qwen2.5-72B-Instruct-Turbo" # $1.2

MISTRAL_SMALL_24B_INSTRUCT_25_01 = "mistralai/Mistral-Small-24B-Instruct-2501" # 0.8

GEMMA_2_INSTRUCT_27B = "google/gemma-2-27b-it" # 0.8

NOUS_HERMES_2_MIXTRAL_8X7B_DPO = "NousResearch/Nous-Hermes-2-Mixtral-8x7B-DPO" # 0.6
#provider
