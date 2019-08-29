class LabelOntonotes:
    person: None
    norp: None
    fac: None
    org: None
    gpe: None
    loc: None
    product: None
    event: None
    work_of_art: None
    law: None
    language: None
    date: None
    time: None
    percent: None
    money: None
    quantity: None
    ordinal: None
    cardinal: None

    def __init__(self, person: None, norp: None, fac: None, org: None, gpe: None, loc: None, product: None, event: None, work_of_art: None, law: None, language: None, date: None, time: None, percent: None, money: None, quantity: None, ordinal: None, cardinal: None) -> None:
        self.person = person
        self.norp = norp
        self.fac = fac
        self.org = org
        self.gpe = gpe
        self.loc = loc
        self.product = product
        self.event = event
        self.work_of_art = work_of_art
        self.law = law
        self.language = language
        self.date = date
        self.time = time
        self.percent = percent
        self.money = money
        self.quantity = quantity
        self.ordinal = ordinal
        self.cardinal = cardinal
