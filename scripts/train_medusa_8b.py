import torch
import torch.nn.functional as F
from src.models import ModelLoader
from src.medusa import MedusaModel
import bitsandbytes as bnb



if __name__ == "__main__":
    loader = ModelLoader("meta-llama/Llama-3.1-8B-Instruct", device="cuda:0")
    loader.load()
    backbone = loader.model
    tokenizer = loader.tokenizer
    medusa = MedusaModel(backbone, num_heads=4)
    medusa.heads.to(device="cuda:1")  # heads stay float32 for stable training; backbone is already float16
    optimizer = bnb.optim.Adam8bit(medusa.heads.parameters(), lr=1e-3)

    TRAIN_TEXT = [
    """The patient arrived at the emergency department complaining of severe chest pain that had gradually worsened over the previous six hours. The pain radiated to the left shoulder and jaw and was associated with sweating, nausea, and mild shortness of breath. Vital signs showed elevated blood pressure and a rapid heart rate. An electrocardiogram was performed immediately and blood samples were sent for cardiac enzyme analysis. The medical team started oxygen therapy, administered aspirin, and prepared the patient for further evaluation and treatment.""",

    """Diabetes mellitus is a chronic metabolic disorder characterized by elevated blood glucose levels resulting from defects in insulin secretion, insulin action, or both. Long term complications include cardiovascular disease, kidney failure, nerve damage, and retinal disease. Effective management requires lifestyle modification, regular monitoring of blood glucose, adherence to medication, and periodic assessment of organ function to reduce the risk of serious complications.""",

    """Magnetic resonance imaging is a noninvasive diagnostic technique widely used in modern medicine to visualize soft tissues, organs, and pathological changes within the human body. The procedure relies on strong magnetic fields and radiofrequency pulses to generate detailed images without exposing patients to ionizing radiation. Radiologists interpret the resulting images to assist clinicians in diagnosing neurological disorders, musculoskeletal injuries, tumors, and a wide range of other medical conditions.""",

    """Pneumonia is an infection of the lungs that may be caused by bacteria, viruses, or fungi. Common symptoms include fever, cough, chest discomfort, fatigue, and difficulty breathing. Diagnosis is based on clinical examination, imaging studies, and laboratory investigations. Treatment depends on the underlying cause and may involve antibiotics, antiviral medications, supportive care, and careful monitoring of respiratory status until the patient recovers completely.""",

    """Hypertension is one of the most prevalent chronic diseases worldwide and is a major risk factor for heart attack, stroke, kidney disease, and heart failure. Early detection through routine screening and appropriate treatment with lifestyle modifications and antihypertensive medications can significantly reduce morbidity and mortality. Patients are encouraged to maintain a healthy diet, engage in regular physical activity, avoid tobacco use, and follow medical advice consistently over time."""
    ]

    epochs = 100
    for epoch in range(epochs):
        epoch_loss = 0.0
        for TEXT in TRAIN_TEXT:
            input_ids = tokenizer(TEXT, return_tensors="pt").input_ids.to("cuda:0")
            head_logits = medusa(input_ids)

            loss = 0.0


            for k in range(len(head_logits)):
                shift = k+1
                logits_k = head_logits[k][:, :-shift, :]
                labels_k = input_ids[:, shift:]
                labels_k = input_ids[:, shift:].to(logits_k.device)`
                loss_k = F.cross_entropy(logits_k.reshape(-1, logits_k.size(-1)), labels_k.reshape(-1))
                loss = loss + (0.8**k)*loss_k

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
        print(f"epoch {epoch}: loss {epoch_loss/len(TRAIN_TEXT):.4f}")
    torch.save(medusa.heads.state_dict(), "medusa_heads_8b.pt")


