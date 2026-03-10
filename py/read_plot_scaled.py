from pynasonde.digisonde.parsers.sao import SaoExtractor

sao = SaoExtractor(
    "../../Individual_Studies/LWS_AGW_TID_Analysis/data/BC840_20170527(147)_SAO.XML",
    extract_time_from_name= False,
        extract_stn_from_name = False,
)
sao.extract_xml()
df = sao.get_scaled_datasets_xml()
df.to_csv("../../Individual_Studies/LWS_AGW_TID_Analysis/data/scaled.csv", index=False, header=True)