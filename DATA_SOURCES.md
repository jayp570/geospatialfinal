# Where to get the real data

Written 2026-11-16 with Claude (Anthropic). Every link below was found by web search on that date. Items marked **not opened** could not be checked by the AI and need a person to click through. Nothing here has been downloaded yet.

**The rule for every file:** it must describe 2020 or earlier, except the permits (the label), which cover 2021–2025. Write the snapshot date of each file you download into the table at the bottom.

## 1. 2020 TCAD appraisal roll (needed for `improvement_value_ratio` and the vacant/underused filter)

**Best source: Texas Data Repository (hosted by UT libraries).**

- Dataset: "Travis Central Appraisal District - 2020 CERTIFIED APPRAISAL ROLL EXPORT"
- Link: https://doi.org/10.18738/T8/QGRX7B
- Its listing says it was downloaded from the TCAD website on Oct 16, 2020. The same collection (https://dataverse.tdl.org/dataverse/ai-housing) also holds the 2019 certified export (https://doi.org/10.18738/T8/SJOR5W).
- **Not opened.** The file list, sizes and whether a UT login is needed are unconfirmed.

Steps:

1. Open the DOI link, download every file (expect one or more large zip files).
2. Unzip. TCAD exports are fixed-width text files (one for properties, others for land, improvements and so on) plus a layout document that gives each field's start position and length.
3. Find the layout document. If the download has none, TCAD posts "Export Layouts" at https://traviscad.org/publicinformation (section "Export layout file explanation"). That is the current layout, so check that its record lengths match the 2020 files before trusting it.
4. Send the layout document (or its first pages) to whoever is running `build_features.py`. The field positions for property id, geographic id, land value and improvement value come from that document. **They are not written into any script yet, because nobody has seen the file.**
5. Convert the property file to a small CSV with four columns: property id, geographic id, land value, improvement value. `build_features.py --appraisal` reads that CSV.

**Why not TCAD directly?** On 2026-11-16 the TCAD public information page listed only 2026 exports. Older years are not posted. If the repository copy fails, file a public information request with TCAD for the "2020 certified appraisal roll export" (same page, "online portal").

## 2. 2020 parcel polygons (needed so permits are matched to the lots that existed in 2020)

The Travis County layer that `pull_parcels.py` downloads is rebuilt monthly. It shows today's lots, so a lot subdivided in 2023 appears as its 2023 pieces. Use it only for the test pull and for finding the id field.

**Best source: TxGIO (formerly TNRIS) StratMap Land Parcels,** the state's yearly collection from every appraisal district.

- Program page: https://www.geographic.texas.gov/stratmap/land-parcels.html
- Download site (DataHub): https://data.geographic.texas.gov/?s=land%20parcels&pg=1
- Field list (PDF): https://www.geographic.texas.gov/_uploads/TxGIO-land-parcel-schema.pdf
- Confirmed to exist as statewide yearly map services: 2019, 2022, 2023, 2024, 2025. A hosted copy of the 2019 collection is labeled "Land Parcels, 2019-03-01".
- **Unconfirmed: whether a 2020 or 2021 collection exists for Travis County.** The DataHub has to be checked by hand.

Steps:

1. Open the DataHub link. Find the "Land Parcels" collections and note every year offered.
2. Open the newest collection **dated 2020 or earlier** and download Travis County (FIPS 48453). The program page says the year and month are in the file name as YYYYMM.
3. Open the attribute table and read the `tax_year` and `date_acq` fields. Those are the snapshot date. Record them.
4. Check which id fields are present and compare them with `tcad_id` in the permits.
5. Convert to GeoJSON or GeoPackage (QGIS: right-click the layer, Export, Save Features As) and name it `parcels_2020.geojson` or `parcels_2020.gpkg`.

Which snapshot is acceptable:

| Snapshot | Use it? | What to say |
|---|---|---|
| 2020 | Yes | Matches the design |
| 2019 | Yes | Before the cutoff, so it cannot leak. Lots created during 2019–2020 are missing or appear as their parent lot |
| 2021 or later | Only as a stated fallback | After the cutoff. Lot shapes may already reflect projects that are part of the label |

The StratMap schema may also carry land, improvement and market values for the snapshot's tax year. If the 2019 or 2020 file has them filled in for Travis County, they could stand in for the appraisal roll. Check before relying on it.

**Fallback:** request the "2020 certified parcel shapefile" from TCAD through the public information portal at https://traviscad.org/publicinformation. Allow several business days.

## 3. Permits (the label) — works today

`python pull_permits.py --since 2021-01-01 --out permits_raw.csv`

Source: City of Austin "Issued Construction Permits", https://data.austintexas.gov/resource/3syk-w9eu.json. A two-row sample on 2026-11-16 returned the expected columns (`tcad_id`, `issue_date`, `permit_class_mapped`, `work_class`, `latitude`, `longitude`). One of the two rows had no coordinates, so some permits will be dropped in location mode. `build_labels.py` prints how many.

## 4. Other layers for `build_features.py`

| Feature | Source | Link | Date note |
|---|---|---|---|
| `zoning_class` | City of Austin "Zoning (Small Map Scale)", field `ZONING_ZTYPE` | https://data.austintexas.gov/d/rwvf-3qkm | **Current zoning only.** No 2020 version was found. State this as a limitation |
| `in_floodplain` | FEMA National Flood Hazard Layer, Travis County (48453C) | https://msc.fema.gov/portal/advanceSearch (**not opened**) | Check the effective date of the panels. Use the flood hazard area polygons |
| `dist_to_road` | Census TIGER/Line 2020 roads, Travis County | https://www2.census.gov/geo/tiger/TIGER2020/ROADS/tl_2020_48453_roads.zip (**not opened**; standard Census file naming) | 2020 vintage |
| Austin city limits | City of Austin jurisdictions layer | Search "jurisdictions" on https://data.austintexas.gov (**link not found yet**) | Keep "AUSTIN FULL PURPOSE"; current boundary |
| `slope`, `elevation` | USGS 3DEP 10 m, through Earth Engine | `ee_extract_embeddings.py --terrain-out` | Terrain. Acquisition dates vary |
| `land_cover_class` | NLCD 2019, through Earth Engine | `ee_extract_embeddings.py --terrain-out` | 2019 |

## 5. Snapshot log (fill this in as files arrive)

| File | Source | Snapshot date | Downloaded by | Downloaded on |
|---|---|---|---|---|
| `permits_raw.csv` | | 2021-01-01 onward | | |
| `parcels_2020.*` | | | | |
| appraisal roll | | | | |
| zoning | | current (limitation) | | |
| floodplain | | | | |
| roads | | 2020 | | |
