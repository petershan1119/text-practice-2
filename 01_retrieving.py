
for paragraphNum, pTag in enumerate(contentPTags):
    # We only want hrefs that link to wiki pages
    tagLinks = pTag.findAll('a', href=re.compile('/wiki/'), class_=False)
    