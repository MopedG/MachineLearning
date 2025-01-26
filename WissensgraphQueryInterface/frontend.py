import streamlit as st


st.title("Abfrage der Kunstshow-Ontologie")
st.subheader("Wissensgraph Query Interface")


q = [
    {
        "vorlage": "Wie viele X hat Y gekauft?",
        "query": [
            "Wie viele",
            {
                "placeholder": "Gemälde"
            },
            "hat",
            {
                "placeholder": " Person"
            },
            "gekauft?"
        ]
    },
    {
        "vorlage": "Aus welchem Land kommt der Account X?",
        "query": [
            "Aus welchem Land kommt der Account",
            { "placeholder": "Accountname" },
            "?"
        ]
    }
]

selection = st.selectbox(
    label="Abfragevorlage",
    options=[template["vorlage"] for template in q]
)

if selection:
    query = None
    for template in q:
        if template["vorlage"] == selection:
            query = template

    columns = st.columns(len(query["query"]), vertical_alignment="bottom", gap="small")
    for i in range(len(query["query"])):
        part = query["query"][i]
        if isinstance(part, dict):
            columns[i].text_input(label=part["placeholder"], label_visibility="hidden", key=part["placeholder"], placeholder=part["placeholder"])
        else:
            columns[i].text(part)






queryAufbau = "WELCHE X HAT Y "
X = "X: bspw. PersonAccount, Contact, Artwork, Ticket, User, Account, Event"
Y = "Y: bspw. Name, Title, TicketType, BillingState, EventName, AccountName"

dict = {
    "query1": "Aus welchem Land kommt der Account 'Wealth Management AG'?",
    "query2": "Welches Artwork hat den Titel 'Golden Statue'?",
    "query3": "Welche ArtworkMedium haben den artMediumTitle 'Photography'?",
    "query4": "Welche Tickets haben den TicketType 'Premium Ticket'?",
    "query5": "Welche PersonAccounts kommen aus dem BillingState 'USA'?",
    "query6": "Welche Contacts haben eine mail unter '@test.com'?"
}



