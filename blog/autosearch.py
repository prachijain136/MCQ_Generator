def autosearch(search):
    try:
        import wikipedia
        return wikipedia.page(search).content
    except:
        return "Maaf Kardo Pls"