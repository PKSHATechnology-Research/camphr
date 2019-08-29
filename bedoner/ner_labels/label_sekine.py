from typing import Dict


class PuneHedgehog:
    pass

    def __init__(self, ) -> None:
        pass


class Color:
    color_other: PuneHedgehog
    nature_color: PuneHedgehog

    def __init__(self, color_other: PuneHedgehog, nature_color: PuneHedgehog) -> None:
        self.color_other = color_other
        self.nature_color = nature_color


class Disease:
    animal_disease: PuneHedgehog
    disease_other: PuneHedgehog

    def __init__(self, animal_disease: PuneHedgehog, disease_other: PuneHedgehog) -> None:
        self.animal_disease = animal_disease
        self.disease_other = disease_other


class Incident:
    incident_other: PuneHedgehog
    war: PuneHedgehog

    def __init__(self, incident_other: PuneHedgehog, war: PuneHedgehog) -> None:
        self.incident_other = incident_other
        self.war = war


class NaturalPhenomenon:
    earthquake: PuneHedgehog
    natural_disaster: PuneHedgehog
    natural_phenomenon_other: PuneHedgehog

    def __init__(self, earthquake: PuneHedgehog, natural_disaster: PuneHedgehog, natural_phenomenon_other: PuneHedgehog) -> None:
        self.earthquake = earthquake
        self.natural_disaster = natural_disaster
        self.natural_phenomenon_other = natural_phenomenon_other


class Occasion:
    conference: PuneHedgehog
    game: PuneHedgehog
    occasion_other: PuneHedgehog
    religious_festival: PuneHedgehog

    def __init__(self, conference: PuneHedgehog, game: PuneHedgehog, occasion_other: PuneHedgehog, religious_festival: PuneHedgehog) -> None:
        self.conference = conference
        self.game = game
        self.occasion_other = occasion_other
        self.religious_festival = religious_festival


class Event:
    event_other: PuneHedgehog
    incident: Incident
    natural_phenomenon: NaturalPhenomenon
    occasion: Occasion

    def __init__(self, event_other: PuneHedgehog, incident: Incident, natural_phenomenon: NaturalPhenomenon, occasion: Occasion) -> None:
        self.event_other = event_other
        self.incident = incident
        self.natural_phenomenon = natural_phenomenon
        self.occasion = occasion


class ArchaeologicalPlace:
    archaeological_place_other: PuneHedgehog
    tumulus: PuneHedgehog

    def __init__(self, archaeological_place_other: PuneHedgehog, tumulus: PuneHedgehog) -> None:
        self.archaeological_place_other = archaeological_place_other
        self.tumulus = tumulus


class Line:
    bridge: PuneHedgehog
    canal: PuneHedgehog
    line_other: PuneHedgehog
    railroad: PuneHedgehog
    road: PuneHedgehog
    tunnel: PuneHedgehog
    water_route: PuneHedgehog

    def __init__(self, bridge: PuneHedgehog, canal: PuneHedgehog, line_other: PuneHedgehog, railroad: PuneHedgehog, road: PuneHedgehog, tunnel: PuneHedgehog, water_route: PuneHedgehog) -> None:
        self.bridge = bridge
        self.canal = canal
        self.line_other = line_other
        self.railroad = railroad
        self.road = road
        self.tunnel = tunnel
        self.water_route = water_route


class Facility:
    archaeological_place: ArchaeologicalPlace
    facility_other: PuneHedgehog
    facility_part: PuneHedgehog
    goe: Dict[str, PuneHedgehog]
    line: Line

    def __init__(self, archaeological_place: ArchaeologicalPlace, facility_other: PuneHedgehog, facility_part: PuneHedgehog, goe: Dict[str, PuneHedgehog], line: Line) -> None:
        self.archaeological_place = archaeological_place
        self.facility_other = facility_other
        self.facility_part = facility_part
        self.goe = goe
        self.line = line


class Address:
    address_other: PuneHedgehog
    email: PuneHedgehog
    phone_number: PuneHedgehog
    postal_address: PuneHedgehog
    url: PuneHedgehog

    def __init__(self, address_other: PuneHedgehog, email: PuneHedgehog, phone_number: PuneHedgehog, postal_address: PuneHedgehog, url: PuneHedgehog) -> None:
        self.address_other = address_other
        self.email = email
        self.phone_number = phone_number
        self.postal_address = postal_address
        self.url = url


class AstralBody:
    astral_body_other: PuneHedgehog
    constellation: PuneHedgehog
    planet: PuneHedgehog
    star: PuneHedgehog

    def __init__(self, astral_body_other: PuneHedgehog, constellation: PuneHedgehog, planet: PuneHedgehog, star: PuneHedgehog) -> None:
        self.astral_body_other = astral_body_other
        self.constellation = constellation
        self.planet = planet
        self.star = star


class GeologicalRegion:
    bay: PuneHedgehog
    geological_region_other: PuneHedgehog
    island: PuneHedgehog
    lake: PuneHedgehog
    mountain: PuneHedgehog
    river: PuneHedgehog
    sea: PuneHedgehog

    def __init__(self, bay: PuneHedgehog, geological_region_other: PuneHedgehog, island: PuneHedgehog, lake: PuneHedgehog, mountain: PuneHedgehog, river: PuneHedgehog, sea: PuneHedgehog) -> None:
        self.bay = bay
        self.geological_region_other = geological_region_other
        self.island = island
        self.lake = lake
        self.mountain = mountain
        self.river = river
        self.sea = sea


class Gpe:
    city: PuneHedgehog
    country: PuneHedgehog
    county: PuneHedgehog
    gpe_other: PuneHedgehog
    province: PuneHedgehog

    def __init__(self, city: PuneHedgehog, country: PuneHedgehog, county: PuneHedgehog, gpe_other: PuneHedgehog, province: PuneHedgehog) -> None:
        self.city = city
        self.country = country
        self.county = county
        self.gpe_other = gpe_other
        self.province = province


class Region:
    continental_region: PuneHedgehog
    domestic_region: PuneHedgehog
    region_other: PuneHedgehog

    def __init__(self, continental_region: PuneHedgehog, domestic_region: PuneHedgehog, region_other: PuneHedgehog) -> None:
        self.continental_region = continental_region
        self.domestic_region = domestic_region
        self.region_other = region_other


class Location:
    address: Address
    astral_body: AstralBody
    geological_region: GeologicalRegion
    gpe: Gpe
    location_other: PuneHedgehog
    region: Region
    spa: PuneHedgehog

    def __init__(self, address: Address, astral_body: AstralBody, geological_region: GeologicalRegion, gpe: Gpe, location_other: PuneHedgehog, region: Region, spa: PuneHedgehog) -> None:
        self.address = address
        self.astral_body = astral_body
        self.geological_region = geological_region
        self.gpe = gpe
        self.location_other = location_other
        self.region = region
        self.spa = spa


class LivingThing:
    amphibia: PuneHedgehog
    bird: PuneHedgehog
    fish: PuneHedgehog
    flora: PuneHedgehog
    fungus: PuneHedgehog
    insect: PuneHedgehog
    living_thing_other: PuneHedgehog
    mammal: PuneHedgehog
    mollusc_arthropod: PuneHedgehog
    reptile: PuneHedgehog

    def __init__(self, amphibia: PuneHedgehog, bird: PuneHedgehog, fish: PuneHedgehog, flora: PuneHedgehog, fungus: PuneHedgehog, insect: PuneHedgehog, living_thing_other: PuneHedgehog, mammal: PuneHedgehog, mollusc_arthropod: PuneHedgehog, reptile: PuneHedgehog) -> None:
        self.amphibia = amphibia
        self.bird = bird
        self.fish = fish
        self.flora = flora
        self.fungus = fungus
        self.insect = insect
        self.living_thing_other = living_thing_other
        self.mammal = mammal
        self.mollusc_arthropod = mollusc_arthropod
        self.reptile = reptile


class LivingThingPart:
    animal_part: PuneHedgehog
    flora_part: PuneHedgehog
    living_thing_part_other: PuneHedgehog

    def __init__(self, animal_part: PuneHedgehog, flora_part: PuneHedgehog, living_thing_part_other: PuneHedgehog) -> None:
        self.animal_part = animal_part
        self.flora_part = flora_part
        self.living_thing_part_other = living_thing_part_other


class NaturalObject:
    compound: PuneHedgehog
    element: PuneHedgehog
    living_thing: LivingThing
    living_thing_part: LivingThingPart
    mineral: PuneHedgehog
    natural_object_other: PuneHedgehog

    def __init__(self, compound: PuneHedgehog, element: PuneHedgehog, living_thing: LivingThing, living_thing_part: LivingThingPart, mineral: PuneHedgehog, natural_object_other: PuneHedgehog) -> None:
        self.compound = compound
        self.element = element
        self.living_thing = living_thing
        self.living_thing_part = living_thing_part
        self.mineral = mineral
        self.natural_object_other = natural_object_other


class Corporation:
    company: PuneHedgehog
    company_group: PuneHedgehog
    corporation_other: PuneHedgehog

    def __init__(self, company: PuneHedgehog, company_group: PuneHedgehog, corporation_other: PuneHedgehog) -> None:
        self.company = company
        self.company_group = company_group
        self.corporation_other = corporation_other


class EthnicGroup:
    ethnic_group_other: PuneHedgehog
    nationality: PuneHedgehog

    def __init__(self, ethnic_group_other: PuneHedgehog, nationality: PuneHedgehog) -> None:
        self.ethnic_group_other = ethnic_group_other
        self.nationality = nationality


class PoliticalOrganization:
    cabinet: PuneHedgehog
    government: PuneHedgehog
    military: PuneHedgehog
    political_organization_other: PuneHedgehog
    political_party: PuneHedgehog

    def __init__(self, cabinet: PuneHedgehog, government: PuneHedgehog, military: PuneHedgehog, political_organization_other: PuneHedgehog, political_party: PuneHedgehog) -> None:
        self.cabinet = cabinet
        self.government = government
        self.military = military
        self.political_organization_other = political_organization_other
        self.political_party = political_party


class SportsOrganization:
    pro_sports_organization: PuneHedgehog
    sports_league: PuneHedgehog
    sports_organization_other: PuneHedgehog

    def __init__(self, pro_sports_organization: PuneHedgehog, sports_league: PuneHedgehog, sports_organization_other: PuneHedgehog) -> None:
        self.pro_sports_organization = pro_sports_organization
        self.sports_league = sports_league
        self.sports_organization_other = sports_organization_other


class Organization:
    corporation: Corporation
    ethnic_group: EthnicGroup
    family: PuneHedgehog
    international_organization: PuneHedgehog
    organization_other: PuneHedgehog
    political_organization: PoliticalOrganization
    show_organization: PuneHedgehog
    sports_organization: SportsOrganization

    def __init__(self, corporation: Corporation, ethnic_group: EthnicGroup, family: PuneHedgehog, international_organization: PuneHedgehog, organization_other: PuneHedgehog, political_organization: PoliticalOrganization, show_organization: PuneHedgehog, sports_organization: SportsOrganization) -> None:
        self.corporation = corporation
        self.ethnic_group = ethnic_group
        self.family = family
        self.international_organization = international_organization
        self.organization_other = organization_other
        self.political_organization = political_organization
        self.show_organization = show_organization
        self.sports_organization = sports_organization


class Art:
    art_other: PuneHedgehog
    book: PuneHedgehog
    broadcast_program: PuneHedgehog
    movie: PuneHedgehog
    music: PuneHedgehog
    picture: PuneHedgehog
    show: PuneHedgehog

    def __init__(self, art_other: PuneHedgehog, book: PuneHedgehog, broadcast_program: PuneHedgehog, movie: PuneHedgehog, music: PuneHedgehog, picture: PuneHedgehog, show: PuneHedgehog) -> None:
        self.art_other = art_other
        self.book = book
        self.broadcast_program = broadcast_program
        self.movie = movie
        self.music = music
        self.picture = picture
        self.show = show


class DoctrineMethod:
    academic: PuneHedgehog
    culture: PuneHedgehog
    doctrine_method_other: PuneHedgehog
    movement: PuneHedgehog
    plan: PuneHedgehog
    religion: PuneHedgehog
    sport: PuneHedgehog
    style: PuneHedgehog
    theory: PuneHedgehog

    def __init__(self, academic: PuneHedgehog, culture: PuneHedgehog, doctrine_method_other: PuneHedgehog, movement: PuneHedgehog, plan: PuneHedgehog, religion: PuneHedgehog, sport: PuneHedgehog, style: PuneHedgehog, theory: PuneHedgehog) -> None:
        self.academic = academic
        self.culture = culture
        self.doctrine_method_other = doctrine_method_other
        self.movement = movement
        self.plan = plan
        self.religion = religion
        self.sport = sport
        self.style = style
        self.theory = theory


class Food:
    dish: PuneHedgehog
    food_other: PuneHedgehog

    def __init__(self, dish: PuneHedgehog, food_other: PuneHedgehog) -> None:
        self.dish = dish
        self.food_other = food_other


class Language:
    language_other: PuneHedgehog
    national_language: PuneHedgehog

    def __init__(self, language_other: PuneHedgehog, national_language: PuneHedgehog) -> None:
        self.language_other = language_other
        self.national_language = national_language


class Printing:
    magazine: PuneHedgehog
    newspaper: PuneHedgehog
    printing_other: PuneHedgehog

    def __init__(self, magazine: PuneHedgehog, newspaper: PuneHedgehog, printing_other: PuneHedgehog) -> None:
        self.magazine = magazine
        self.newspaper = newspaper
        self.printing_other = printing_other


class Rule:
    law: PuneHedgehog
    rule_other: PuneHedgehog
    treaty: PuneHedgehog

    def __init__(self, law: PuneHedgehog, rule_other: PuneHedgehog, treaty: PuneHedgehog) -> None:
        self.law = law
        self.rule_other = rule_other
        self.treaty = treaty


class Title:
    position_vocation: PuneHedgehog
    title_other: PuneHedgehog

    def __init__(self, position_vocation: PuneHedgehog, title_other: PuneHedgehog) -> None:
        self.position_vocation = position_vocation
        self.title_other = title_other


class Unit:
    currency: PuneHedgehog
    unit_other: PuneHedgehog

    def __init__(self, currency: PuneHedgehog, unit_other: PuneHedgehog) -> None:
        self.currency = currency
        self.unit_other = unit_other


class Vehicle:
    aircraft: PuneHedgehog
    car: PuneHedgehog
    ship: PuneHedgehog
    spaceship: PuneHedgehog
    train: PuneHedgehog
    vehicle_other: PuneHedgehog

    def __init__(self, aircraft: PuneHedgehog, car: PuneHedgehog, ship: PuneHedgehog, spaceship: PuneHedgehog, train: PuneHedgehog, vehicle_other: PuneHedgehog) -> None:
        self.aircraft = aircraft
        self.car = car
        self.ship = ship
        self.spaceship = spaceship
        self.train = train
        self.vehicle_other = vehicle_other


class Product:
    art: Art
    award: PuneHedgehog
    character: PuneHedgehog
    product_class: PuneHedgehog
    clothing: PuneHedgehog
    decoration: PuneHedgehog
    doctrine_method: DoctrineMethod
    drug: PuneHedgehog
    food: Food
    id_number: PuneHedgehog
    language: Language
    material: PuneHedgehog
    money_form: PuneHedgehog
    offence: PuneHedgehog
    printing: Printing
    product_other: PuneHedgehog
    rule: Rule
    service: PuneHedgehog
    stock: PuneHedgehog
    title: Title
    unit: Unit
    vehicle: Vehicle
    weapon: PuneHedgehog

    def __init__(self, art: Art, award: PuneHedgehog, character: PuneHedgehog, product_class: PuneHedgehog, clothing: PuneHedgehog, decoration: PuneHedgehog, doctrine_method: DoctrineMethod, drug: PuneHedgehog, food: Food, id_number: PuneHedgehog, language: Language, material: PuneHedgehog, money_form: PuneHedgehog, offence: PuneHedgehog, printing: Printing, product_other: PuneHedgehog, rule: Rule, service: PuneHedgehog, stock: PuneHedgehog, title: Title, unit: Unit, vehicle: Vehicle, weapon: PuneHedgehog) -> None:
        self.art = art
        self.award = award
        self.character = character
        self.product_class = product_class
        self.clothing = clothing
        self.decoration = decoration
        self.doctrine_method = doctrine_method
        self.drug = drug
        self.food = food
        self.id_number = id_number
        self.language = language
        self.material = material
        self.money_form = money_form
        self.offence = offence
        self.printing = printing
        self.product_other = product_other
        self.rule = rule
        self.service = service
        self.stock = stock
        self.title = title
        self.unit = unit
        self.vehicle = vehicle
        self.weapon = weapon


class LabelSekine:
    color: Color
    disease: Disease
    event: Event
    facility: Facility
    god: PuneHedgehog
    location: Location
    name_other: PuneHedgehog
    natural_object: NaturalObject
    organization: Organization
    person: PuneHedgehog
    product: Product

    def __init__(self, color: Color, disease: Disease, event: Event, facility: Facility, god: PuneHedgehog, location: Location, name_other: PuneHedgehog, natural_object: NaturalObject, organization: Organization, person: PuneHedgehog, product: Product) -> None:
        self.color = color
        self.disease = disease
        self.event = event
        self.facility = facility
        self.god = god
        self.location = location
        self.name_other = name_other
        self.natural_object = natural_object
        self.organization = organization
        self.person = person
        self.product = product
