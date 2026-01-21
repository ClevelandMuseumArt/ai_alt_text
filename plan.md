# Alt Text Generation Plan
## Step 1 - ✅
### Date: N/A
- Attain Google Console Credits
- Confirm access to Google Console
- Sign Piction SOW for update process
- Create appropriate Piction metadata tags
  -  ALT_TEXT_MEETS_THRESHOLD
    - can either be:
      - TRUE
      - FALSE
      - OVERWRITTEN
## Step 2 - ❌
### Date: Week of 12/15
- Finetune Gemini (3 / 2.5) on initial train data
- Implement rough draft of agentic script
- Ensure Piction is continuing work on exposing PUT / update endpoint and auto update of metadata tag when alt text is manually edited (ALT_TEXT_MEETS_THRESHOLD)
- Ensure metadata tag is set appropriately for all manually written alt texts (ALT_TEXT_MEETS_THRESHOLD)
- Make Jim aware of necessary CO-API updates needed
  - human_reviewed property is set to false unless ALT_TEXT_MEETS_THRESHOLD equals OVERWRITTEN
## Step 3 - ❌
### Date: Week of 12/22
- Generate alt text for all primary images that have no current alt text
  - any images without the ALT_TEXT_MEETS_THRESHOLD metatag equal to OVERWRITTEN
- Run analytics / metrics on results
  - how many hit minimum threshold, etc
- Determine if manaual review / further fine tuning / prompting is needed
  - Revise minimum quantative thresholds as needed
- Test / implement CO-API updates on develop branch
- FE implementation on dev / staging after CO-API updates implemented
## Step 4 - ❌
### Date: Week of 12/29
- Rerun alt text generation for all primary images in piction without ALT_TEXT_MEETS_THRESHOLD metatag equal to OVERWRITTEN
- Upload initial data to Piction via CSV metadata upload process
- Implement CO-API updates on production
- Expose on initial implementation on production FE site
- Test manual update of alt text in Piction process
  - Ensure metadata tag is changed to OVERWRITTEN and CO-API / website FE update accordingly
## Step 5 -  ❌
### Date: Week of 1/5
- Implement Piction update process using exposed endpoint and CRON server script
  - Ensure that it's possible to complete a backlog data infill from previous month's data
- Monitor processes as needed

<!--


Generate alt-text for subset of primary images (Tara / Sam) using all methods
  - choose a few hundred diverse artworks w/o alt-text to run through the model as a POC
3. Manual review of subset of alt text (Delaney / Andrea / Tara / Sam)
  - Delaney will review a portion of the alt-texts before we proceed with a more general review
  - use exsiting 1-5 scale
4. Revise minimum thresholds for cosine similarity and inner product array, which will dictate when an alt-text does (not) need review (Tara / Sam)
5. Run initial alt-text generation for 65k+ primary images (Tara / Sam)
  - confirm quantative measurement alignment on most results
6. Expose form via FE augmentation to allow for community input (Jeff / Tara / Sam)
  - implemented alongside form refactor
  - confirm language
7. Upload data to piction via csv bulk upload (Tara / Sam / Andrea)
  - run initial script
  - ensure that majority of alt texts are passing quantative measures
  - if necessary re-evaluate prompts / finetuning
8. Create script that is run via either shell command / CRON timer on piction server / linux vm (Tara / Sam)
  - CRON job will run daily @ 6:00 pm
  - piction update logs / api endpoint will be crawled for image information that needs to have alt text generated
    - UMO id, image url, updated at, ...
  - creation of metadata.json will signal that images are ready to be updated
    - metadata will include rows headed by [UMO, alt_text, meets_threshold]
    - signal sent to piction via API call
    - will be controlled by a BE data field (read only in UI)
    - any edits to alt text in UI will automatically mark the meets_threshold field as "OVERWRITTEN"
    - otherwise meets_threshold will be "TRUE" or "FALSE"
9. Add script to piction server / linux vm (Tara / Sam / Andrea)
  - will flag image alt texts that need human review / edits post piction upload / ingestion
    - controlled by meets_threshold field
    - queries will be run on a (weekly / bi-weekly) schedule to identify alt_texts that need to be reviewed by content specialist
      - can be achieved using an advanced search piction link
  - generate alt text ad hoc strategy / command line prompt to run alt text generation on image uploads folder (confirm strategy)
    - runs log crawl / alt text generation script
  - confirm expected metadata file format
    - json / csv
  - piction will send signal to athena which will in turn signal CO-API to update artwork
10. Update CO-API (Jim Kilmer / Sam / Tara / Jeff)
  - will either search for piction flag
  - mark the alt text as human_reviewed "false" (YES || NO meets_threshold value) or "true" (OVERWRITTENT meets_threshold value)
  - will determine the language disclaimer on the FE of the website
  - not derivative dependent (one alt text per image)
11. Monitor as necessary (Tara / Sam / Andrea / Delaney)
  - metrics on number of image alt texts "flagged" / "edited" by community
  - creation of new email distribution list specifically for ai alt text flagging (e.g. alt_text@clevelandart.org) -->
