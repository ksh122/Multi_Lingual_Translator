import pandas as pd

# Define dataset with English sentences and their translations
data = {
    "English": [
        "The Earth revolves around the Sun.",
        "Water boils at 100 degrees Celsius.",
        "Mahatma Gandhi led the freedom movement in India.",
        "The Pythagorean theorem states that a² + b² = c².",
        "Shakespeare wrote many famous plays.",
        "Photosynthesis is the process by which plants make food.",
        "The human body has 206 bones.",
        "Newton's laws explain the motion of objects.",
        "The capital of India is New Delhi.",
        "Electrons are negatively charged particles.",
        "The Taj Mahal is a symbol of love.",
        "A verb is an action word in a sentence.",
        "The Indus Valley Civilization was one of the earliest urban cultures.",
        "A triangle has three sides.",
        "Water is essential for all living beings.",
        "The brain controls the functions of the human body.",
        "The Himalayas are the highest mountain range in the world.",
        "Democracy is a form of government by the people.",
        "The periodic table organizes elements based on their properties.",
        "Gravity is the force that pulls objects toward the Earth."
    ],
    "Hindi": [
        "पृथ्वी सूर्य के चारों ओर घूमती है।",
        "पानी 100 डिग्री सेल्सियस पर उबलता है।",
        "महात्मा गांधी ने भारत में स्वतंत्रता आंदोलन का नेतृत्व किया।",
        "पाइथागोरस प्रमेय कहती है कि a² + b² = c²।",
        "शेक्सपियर ने कई प्रसिद्ध नाटक लिखे।",
        "प्रकाश संश्लेषण वह प्रक्रिया है जिससे पौधे भोजन बनाते हैं।",
        "मानव शरीर में 206 हड्डियाँ होती हैं।",
        "न्यूटन के नियम वस्तुओं की गति को समझाते हैं।",
        "भारत की राजधानी नई दिल्ली है।",
        "इलेक्ट्रॉन नकारात्मक रूप से आवेशित कण होते हैं।",
        "ताज महल प्रेम का प्रतीक है।",
        "क्रिया एक वाक्य में क्रिया शब्द होता है।",
        "सिंधु घाटी सभ्यता प्रारंभिक शहरी संस्कृतियों में से एक थी।",
        "एक त्रिभुज में तीन भुजाएँ होती हैं।",
        "पानी सभी जीवित प्राणियों के लिए आवश्यक है।",
        "मस्तिष्क मानव शरीर के कार्यों को नियंत्रित करता है।",
        "हिमालय दुनिया की सबसे ऊँची पर्वत श्रृंखला है।",
        "लोकतंत्र जनता द्वारा संचालित सरकार का एक रूप है।",
        "आवर्त सारणी तत्वों को उनकी विशेषताओं के आधार पर व्यवस्थित करती है।",
        "गुरुत्वाकर्षण वह बल है जो वस्तुओं को पृथ्वी की ओर खींचता है।"
    ],
    "Gujarati": [
        "પૃથ્વી સૂર્યની આજુબાજુ ફરે છે.",
        "પાણી 100 ડિગ્રી સેલ્સિયસ પર ઉકળે છે.",
        "મહાત્મા ગાંધી એ ભારતના સ્વતંત્રતા આંદોલનનું નેતૃત્વ કર્યું.",
        "પાઈથાગોરસ સિદ્ધાંત કહે છે કે a² + b² = c².",
        "શેક્સપિયર એ ઘણા પ્રસિદ્ધ નાટકો લખ્યા.",
        "પ્રકાશ સંશ્લેષણ એ પ્રક્રિયા છે જેના દ્વારા વનસ્પતિઓ ખોરાક બનાવે છે.",
        "માનવ શરીરમાં 206 હાડકાં હોય છે.",
        "ન્યુટનનાં નિયમો વસ્તુઓની ગતિને સમજાવે છે.",
        "ભારતની રાજધાની નવી દિલ્હી છે.",
        "ઇલેક્ટ્રોન્સ ઋણ ચાર્જ ધરાવતા કણો છે.",
        "તાજ મહેલ પ્રેમનું પ્રતીક છે.",
        "ક્રિયા એક વાક્યમાં ક્રિયાપદ છે.",
        "સિંધુ ખીણ સંસ્કૃતિ પ્રથમ શહેરી સંસ્કૃતિઓમાંની એક હતી.",
        "ત્રિકોણમાં ત્રણ બાજુઓ હોય છે.",
        "પાણી તમામ જીવંત જીવો માટે આવશ્યક છે.",
        "મગજ માનવ શરીરના કાર્યોને નિયંત્રિત કરે છે.",
        "હિમાલય વિશ્વની સૌથી ઊંચી પર્વત શ્રેણી છે.",
        "પ્રજાસત્તાક એ લોકો દ્વારા સંચાલિત શાસન પ્રણાલી છે.",
        "આવર્ત કોષ્ટક તત્વોને તેમની ગુણધર્મોના આધારે ગોઠવે છે.",
        "ગુરુત્વાકર્ષણ એ શક્તિ છે જે વસ્તુઓને પૃથ્વી તરફ ખેંચે છે."
    ],
    "Marathi": [
        "पृथ्वी सूर्याभोवती फिरते.",
        "पाणी 100 अंश सेल्सियस तापमानाला उकळते.",
        "महात्मा गांधींनी भारताच्या स्वातंत्र्य चळवळीचे नेतृत्व केले.",
        "पायथागोरसचा सिद्धांत सांगतो की a² + b² = c².",
        "शेक्सपियरने अनेक प्रसिद्ध नाटके लिहिली.",
        "प्रकाशसंश्लेषण ही वनस्पती अन्न तयार करण्याची प्रक्रिया आहे.",
        "मानवी शरीरात 206 हाडे असतात.",
        "न्यूटनचे नियम वस्तूंची गती स्पष्ट करतात.",
        "भारताची राजधानी नवी दिल्ली आहे.",
        "इलेक्ट्रॉन्स नकारात्मक आवेशित कण असतात.",
        "ताजमहाल प्रेमाचे प्रतीक आहे.",
        "क्रियापद हे वाक्यातील क्रियाशब्द आहे.",
        "सिंधू संस्कृती ही प्राचीन शहरी संस्कृतींपैकी एक होती.",
        "त्रिकोणाला तीन बाजू असतात.",
        "पाणी सर्व सजीवांसाठी आवश्यक आहे.",
        "मेंदू मानवी शरीराचे कार्य नियंत्रित करतो.",
        "हिमालय ही जगातील सर्वात उंच पर्वतरांग आहे.",
        "लोकशाही ही लोकांच्या द्वारे चालवली जाणारी शासनपद्धती आहे.",
        "आवर्त सारणी तत्त्वांना त्यांच्या गुणधर्मांनुसार संघटित करते.",
        "गुरुत्वाकर्षण हे पृथ्वीच्या दिशेने वस्तूंना आकर्षित करणारे बल आहे."
    ]
}

# Convert into a DataFrame
df = pd.DataFrame(data)

# # Save as CSV
# df.to_csv("educational_translation_dataset.csv", index=False)

# # Display first few rows
# df.head()