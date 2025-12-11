from train_model import process_data

def test_basic_main_content_():
    html = """
    <html>
        <body>
            <header>This is header</header>
            <menu>
                <a href="#">Home</a>
                <a href="#">Products</a>
                <a href="#">About</a>
                <a href="#">Contact</a>

            </menu>
            <main>
                <div class="product">Product 1 <p>Some log decription of some random product bla bla</p><img></div>
                <div class="product">Product 2 <p>Some log decription of some random product bla bla</p><img></div>
                <div class="product">Product 3 <p>Some log decription of some random product bla bla</p><img></div>
                <div class="product">Product 4<p>Some log decription of some random product bla bla</p><img></div>
                <div class="product">Product 5<p>Some log decription of some random product bla bla bla bla</p><img></div>       
                <div class="product">Product 6 <p>Some log decription of some random product bla bla bla bla</p><img></div>
            </main>
            <hr>
            <br>
            <footer>This is footer</footer>
        </body>
    </html>
    """
    main_content = process_data.get_main_html_content_tag(html)
    assert main_content is not None
    assert main_content.tag == 'main'
    products = main_content.findall('.//div[@class="product"]')
    assert len(products) == 6
    assert products[0].text_content() == 'Product 1 Some log decription of some random product bla bla'
    assert products[1].text_content() == 'Product 2 Some log decription of some random product bla bla'


def test_no_main_content():    
    html = """
    <html>
        <style>body {font-family: Arial;}</style>
        <body></body>
        <iframe src="ads.html"></iframe>
        <script>console.log("Hello World")</script>
    </html>
    """

    main_content = process_data.get_main_html_content_tag(html)
    assert main_content is not None
    # print("Main content tag:", main_content.tag, "with attributes:", main_content.attrib, "and classes:", list(main_content.classes), main_content.getchildren()[0].tag)
    assert main_content.tag == 'body'  # Fallback to entire document

def main_content_page(name, expected_tag, expected_class, id = None):
    with open(f'tests/test_data/{name}.html', 'r', encoding='utf-8') as f:
        html = f.read()
    
    main_content = process_data.get_main_html_content_tag(html)
    # print(f'Testing {name}: found tag {main_content.tag} with id {main_content.get("id")} and classes {list(main_content.classes)}')

    assert main_content is not None
    assert main_content.tag == expected_tag
    
    
    if id:
        assert main_content.get('id') == id
    if expected_class:
        assert expected_class in list(main_content.classes)

def test_main_content_pages():
    GROUND_TRUTH = {
    'page_1': ('div', 'w3-row-padding', None),
    'page_2': ('div', None, 'content'),
    'page_3': ('div', None, 'content'),
    'page_4': ('div', 'items-list', None),
    'page_5': ('div', None, 'content'),
    'page_6': ('ul', 'products', None),
    'page_7': ('div', None, 'content'),
    'page_8': ('div', None, 'content'),
    'page_9': ('div', None, 'content'),
    'page_10': ('ul', 'list-style-none', None),
    'page_11': ('div', None, 'content'),
    'page_12': ('ul', None, None),
    'page_13': ('div', None, None),
    'page_14': ('div', None, 'incenterpage'),
    'page_15': ('div', 'tab-content', None),

}
    for page, (tag, cls, id) in GROUND_TRUTH.items():
        main_content_page(page, tag, cls, id)