import streamlit as st        
from st_clickable_images import clickable_images
import matplotlib.pyplot as plt
import numpy as np

# This final project proposal template provides an example of product recommendation application for Cornell Tech Shoe Boutique, a family friendly online shoe store. 
# The layout has tabs (instead of sidebar). The tabs include 
# - Cornell Tech Show Boutique (Home page):
# - Women's Shoes, Men's Shoes: Shop for women's shoes, select a recommendation system that's right for you which updates the results (result updates not fully implemented)
# - Recommendation Analysis: Shoes classification performance on dummy data including precision, recall, TP, TN, FP, and FN


def apply_threshold(probabilities, threshold):
    # +1 if >= threshold and -1 otherwise.
    return np.array([1 if p[1] >= threshold else 0 for p in probabilities])
    
tab1, tab2, tab3, tab4 = st.tabs(["Cornell Tech Shoe Boutique", "Women's Shoes", "Men's Shoes", "Recommendation Analysis"])

with tab1:
    st.header("Cornell Tech Shoe Boutique") 
    st.markdown('## Family & Friends Online Shoe Store')
    st.markdown('### What are you shopping for today?')

    categories = []
    
    # Shop by category
    category_select = clickable_images(
        [
            "https://slimages.macysassets.com/is/image/McomMedia/media/041822_MENS_SHOES_CAT_PAGE_01_1460763.jpg?fmt=webp&wid=750",
            "https://slimages.macysassets.com/is/image/McomMedia/media/041822_MENS_SHOES_CAT_PAGE_02_1460764.jpg?fmt=webp&wid=750",
            "https://slimages.macysassets.com/is/image/McomMedia/media/041822_MENS_SHOES_CAT_PAGE_03_1460765.jpg?fmt=webp&wid=750",
            "https://slimages.macysassets.com/is/image/McomMedia/media/041822_MENS_SHOES_CAT_PAGE_06_1460769.jpg?fmt=webp&wid=750",
            "https://slimages.macysassets.com/is/image/McomMedia/media/041822_MENS_SHOES_CAT_PAGE_04_1460766.jpg?fmt=webp&wid=750",
            "https://slimages.macysassets.com/is/image/McomMedia/media/041822_MENS_SHOES_CAT_PAGE_05_1460767.jpg?fmt=webp&wid=750",
        ],
        titles=categories,
        div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
        img_style={"margin": "5px", "height": "100px"},
    )

    st.markdown('### What are you shopping for today?')

    st.write('We are a friends and family online shoe boutique affiliated with Cornell Tech in New York City. We offer name brand shoes at low prices. Using machine learning recommendation systems, we find the right shoe for you! Get prepared with warm-weather styles that are back and better than ever.')


    # Display Cornell Tech Image
    st.image("https://www.som.com/wp-content/uploads/fly-images/8219/cornellnyctech_680x510_som_kilograph_03-1366x1024-c.jpg",
             width=700)   

with tab2:
    st.header("Women's Shoes")
    st.markdown('## Shop by Category')

    women_categories = ['Scandals','Flats','Heels','Sneakers']
    
    # Shop by category
    women_category_select = clickable_images(
        [
            "https://slimages.macysassets.com/is/image/McomMedia/media/022723_CC_SHOE_CAT_PAGE_REFRESH_01_1493681.jpg?fmt=webp&wid=318",
            "https://slimages.macysassets.com/is/image/McomMedia/media/022723_CC_SHOE_CAT_PAGE_REFRESH_03_1493683.jpg?fmt=webp&wid=318",
            "https://slimages.macysassets.com/is/image/McomMedia/media/022723_CC_SHOE_CAT_PAGE_REFRESH_04_1493684.jpg?fmt=webp&wid=318",
            "https://slimages.macysassets.com/is/image/McomMedia/media/022723_CC_SHOE_CAT_PAGE_REFRESH_02_1493682.jpg?fmt=webp&wid=318",
        ],
        titles=women_categories,
        div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
        img_style={"margin": "5px", "height": "200px"},
    )

    st.markdown(f"Category {women_categories[women_category_select]} Selected" if women_category_select > -1 else "")

    st.markdown('### Select a Recommendation System')

    method_col1, method_col2, method_col3 = st.columns([1,1,1])

    method_select=-1
    with(method_col1):
        st.write("Recommendation 1")
        if st.button('Select 1'):
            st.write('Recommendation 1 Selected')
            method_select=1
    with(method_col2):
        st.write("Recommendation 2")
        if st.button('Select 2'):
            st.write('Recommendation 2 Selected')
            method_select=2
    with(method_col3):
        st.write("Recommendation 3")
        if st.button('Select 3'):
            st.write('Recommendation 3 Selected')
            method_select=3
    
    st.markdown('### Your Recommendations')

    if(women_category_select==0):
        # 1 - Flats
        scandals_list=['https://slimages.macysassets.com/is/image/MCY/products/8/optimized/11946438_fpx.tif?op_sharpen=1&wid=700&hei=855&fit=fit,1&fmt=webp',
                    'https://slimages.macysassets.com/is/image/MCY/products/1/optimized/16394021_fpx.tif?op_sharpen=1&wid=700&hei=855&fit=fit,1&fmt=webp',
                    'https://slimages.macysassets.com/is/image/MCY/products/0/optimized/23650170_fpx.tif?op_sharpen=1&wid=700&hei=855&fit=fit,1&fmt=webp',
                    'https://slimages.macysassets.com/is/image/MCY/products/9/optimized/24107519_fpx.tif?op_sharpen=1&wid=700&hei=855&fit=fit,1&fmt=webp',
                    ]
        scandals_list=[
                        "Women's Bennia Thong Sandals",
                        "Women's Kendrick Espadrille Wedge Sandals",
                        "Women's Hadyn Slide Sandals",
                        "Women's Tashia Cutout Logo Slide Sandals"
                    ]
        
        scandals_prices=[35.40,55.82,55.30,55.30]
        
        scandal_select = clickable_images(
            [
                'https://slimages.macysassets.com/is/image/MCY/products/8/optimized/11946438_fpx.tif?op_sharpen=1&wid=700&hei=855&fit=fit,1&fmt=webp',
                'https://slimages.macysassets.com/is/image/MCY/products/1/optimized/16394021_fpx.tif?op_sharpen=1&wid=700&hei=855&fit=fit,1&fmt=webp',
                'https://slimages.macysassets.com/is/image/MCY/products/0/optimized/23650170_fpx.tif?op_sharpen=1&wid=700&hei=855&fit=fit,1&fmt=webp',
                'https://slimages.macysassets.com/is/image/MCY/products/9/optimized/24107519_fpx.tif?op_sharpen=1&wid=700&hei=855&fit=fit,1&fmt=webp',
            ],
            titles=scandals_list,
            div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
            img_style={"margin": "5px", "height": "200px"},
        )

        col1, col2, col3, col4 = st.columns([1,1,1,1])

        with(col1):
            st.write(scandals_list[0])
            st.write(f"${scandals_prices[0]}")
        with(col2):
            st.write(scandals_list[1])
            st.write(f"${scandals_prices[1]}")
        with(col3):
            st.write(scandals_list[2])
            st.write(f"${scandals_prices[2]}")
        with(col4):
            st.write(scandals_list[3])
            st.write(f"${scandals_prices[3]}")

        st.markdown(f"Scandal {scandals_list[scandal_select]} Selected" if scandal_select > -1 else "")

    elif(women_category_select==1):
        flats_list=['https://slimages.macysassets.com/is/image/MCY/products/3/optimized/18380753_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                    'https://slimages.macysassets.com/is/image/MCY/products/0/optimized/24186860_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                    'https://slimages.macysassets.com/is/image/MCY/products/3/optimized/23524353_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                    'https://slimages.macysassets.com/is/image/MCY/products/2/optimized/23808212_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                    ]
        flats_list=[
                        "Balica Loafers",
                        "Women's Belinda Cap-Toe Slingback Flats",
                        "Step N' Flex Women's Neptoon Square-Toe Flats, Created for Macy's",
                        "Women's Ganimay Classic Ballet Flats"
                    ]
        
        flats_prices=[84.00,62.30,41.70,48.30]
        
        flats_select = clickable_images(
            [
                'https://slimages.macysassets.com/is/image/MCY/products/3/optimized/18380753_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                'https://slimages.macysassets.com/is/image/MCY/products/0/optimized/24186860_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                'https://slimages.macysassets.com/is/image/MCY/products/3/optimized/23524353_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                'https://slimages.macysassets.com/is/image/MCY/products/2/optimized/23808212_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
            ],
            titles=flats_list,
            div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
            img_style={"margin": "5px", "height": "200px"},
        )

        flat_col1, flat_col2, flat_col3, flat_col4 = st.columns([1,1,1,1])

        with(flat_col1):
            st.write(flats_list[0])
            st.write(f"${flats_prices[0]}")
        with(flat_col2):
            st.write(flats_list[1])
            st.write(f"${flats_prices[1]}")
        with(flat_col3):
            st.write(flats_list[2])
            st.write(f"${flats_prices[2]}")
        with(flat_col4):
            st.write(flats_list[3])
            st.write(f"${flats_prices[3]}")

        st.markdown(f"Flat {flats_list[flats_select]} Selected" if flats_select > -1 else "")
        # 2 - Heels
        # 3 - Sneakers
    elif(women_category_select==2):
        # 0 - Scandels
        scandals_list=['https://slimages.macysassets.com/is/image/MCY/products/8/optimized/11946438_fpx.tif?op_sharpen=1&wid=700&hei=855&fit=fit,1&fmt=webp',
                    'https://slimages.macysassets.com/is/image/MCY/products/1/optimized/16394021_fpx.tif?op_sharpen=1&wid=700&hei=855&fit=fit,1&fmt=webp',
                    'https://slimages.macysassets.com/is/image/MCY/products/0/optimized/23650170_fpx.tif?op_sharpen=1&wid=700&hei=855&fit=fit,1&fmt=webp',
                    'https://slimages.macysassets.com/is/image/MCY/products/9/optimized/24107519_fpx.tif?op_sharpen=1&wid=700&hei=855&fit=fit,1&fmt=webp',
                    ]
        scandals_list=[
                        "Women's Bennia Thong Sandals",
                        "Women's Kendrick Espadrille Wedge Sandals",
                        "Women's Hadyn Slide Sandals",
                        "Women's Tashia Cutout Logo Slide Sandals"
                    ]
        
        scandals_prices=[35.40,55.82,55.30,55.30]
        
        scandal_select = clickable_images(
            [
                'https://slimages.macysassets.com/is/image/MCY/products/8/optimized/11946438_fpx.tif?op_sharpen=1&wid=700&hei=855&fit=fit,1&fmt=webp',
                'https://slimages.macysassets.com/is/image/MCY/products/1/optimized/16394021_fpx.tif?op_sharpen=1&wid=700&hei=855&fit=fit,1&fmt=webp',
                'https://slimages.macysassets.com/is/image/MCY/products/0/optimized/23650170_fpx.tif?op_sharpen=1&wid=700&hei=855&fit=fit,1&fmt=webp',
                'https://slimages.macysassets.com/is/image/MCY/products/9/optimized/24107519_fpx.tif?op_sharpen=1&wid=700&hei=855&fit=fit,1&fmt=webp',
            ],
            titles=scandals_list,
            div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
            img_style={"margin": "5px", "height": "200px"},
        )

        col1, col2, col3, col4 = st.columns([1,1,1,1])

        with(col1):
            st.write(scandals_list[0])
            st.write(f"${scandals_prices[0]}")
        with(col2):
            st.write(scandals_list[1])
            st.write(f"${scandals_prices[1]}")
        with(col3):
            st.write(scandals_list[2])
            st.write(f"${scandals_prices[2]}")
        with(col4):
            st.write(scandals_list[3])
            st.write(f"${scandals_prices[3]}")

        st.markdown(f"Scandal {scandals_list[scandal_select]} Selected" if scandal_select > -1 else "")

    elif(women_category_select==3):
        sneaker_urls=['https://slimages.macysassets.com/is/image/MCY/products/6/optimized/23675776_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                    'https://slimages.macysassets.com/is/image/MCY/products/1/optimized/23537471_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                    'https://slimages.macysassets.com/is/image/MCY/products/4/optimized/15662224_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                    'https://slimages.macysassets.com/is/image/MCY/products/2/optimized/21019832_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                    ]
        sneaker_list=[
                        "Women's Kinetic Impact II Lace-Up Mesh Sneakers",
                        "Women's Alondra Casual Platform Lace-up Sneakers",
                        "Women's Swift Run Casual Sneakers from Finish Line",
                        "Women's Chuck Taylor Madison Low Top Casual Sneakers from Finish Line"
                    ]
        
        sneaker_prices=[145.00,63.30,70.00,60.00]
        
        sneaker_select = clickable_images(
            [
                'https://slimages.macysassets.com/is/image/MCY/products/6/optimized/23675776_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                'https://slimages.macysassets.com/is/image/MCY/products/1/optimized/23537471_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                'https://slimages.macysassets.com/is/image/MCY/products/4/optimized/15662224_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                'https://slimages.macysassets.com/is/image/MCY/products/2/optimized/21019832_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
            ],
            titles=sneaker_list,
            div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
            img_style={"margin": "5px", "height": "200px"},
        )

        sneaker_col1, sneaker_col2, sneaker_col3, sneaker_col4 = st.columns([1,1,1,1])

        with(sneaker_col1):
            st.write(sneaker_list[0])
            st.write(f"${sneaker_prices[0]}")
        with(sneaker_col2):
            st.write(sneaker_list[1])
            st.write(f"${sneaker_prices[1]}")
        with(sneaker_col3):
            st.write(sneaker_list[2])
            st.write(f"${sneaker_prices[2]}")
        with(sneaker_col4):
            st.write(sneaker_list[3])
            st.write(f"${sneaker_prices[3]}")

        st.markdown(f"Sneaker {sneaker_list[sneaker_select]} Selected" if sneaker_select > -1 else "")

with tab3:
    st.header("Men's Shoes")
    st.markdown('## Shop by Category')

    men_categories = ['Althetic Shoes & Sneakers','Causal Shoes','Dress Shoes','Loafers']
    
    # Shop by category
    men_category_select = clickable_images(
        [
            "https://slimages.macysassets.com/is/image/McomMedia/media/MensShoesAthleticShoesSneakers12114859_1490895.png",
            "https://slimages.macysassets.com/is/image/McomMedia/media/MensShoesCasualShoes21083139_1482542.jpg",
            "https://slimages.macysassets.com/is/image/McomMedia/media/MensShoesDressShoes8843494_1490897.png",
            "https://slimages.macysassets.com/is/image/McomMedia/media/MensShoesLoafers1981573_1490898.png",
        ],
        titles=men_categories,
        div_style={"display": "flex", "justify-content": "left", "flex-wrap": "wrap"},
        img_style={"margin": "5px", "height": "150px"},
    )
    men_col1, men_col2, men_col3, men_col4 = st.columns([1,1,1,1])

    with(men_col1):
        st.write(men_categories[0])
    with(men_col2):
        st.write(men_categories[1])
    with(men_col3):
        st.write(men_categories[2])
    with(men_col4):
        st.write(men_categories[3])

    st.markdown(f"Category {men_categories[men_category_select]} Selected" if men_category_select > -1 else "")

    st.markdown('### Select a Recommendation System')

    men_method_col1, men_method_col2, men_method_col3 = st.columns([1,1,1])

    men_method_select=-1
    with(men_method_col1):
        st.write("Recommendation 1")
        if st.button('Select 1 for Men'):
            st.write('Recommendation 1 Selected')
            men_method_select=1
    with(men_method_col2):
        st.write("Recommendation 2")
        if st.button('Select 2 for Men'):
            st.write('Recommendation 2 Selected')
            men_method_select=2
    with(men_method_col3):
        st.write("Recommendation 3")
        if st.button('Select 3 for Men'):
            st.write('Recommendation 3 Selected')
            men_method_select=3
    
    st.markdown('### Your Recommendations')

    if(men_category_select==0):
        # 1 - Flats
        althetic_urls=['https://slimages.macysassets.com/is/image/MCY/products/7/optimized/21363567_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                    'https://slimages.macysassets.com/is/image/MCY/products/6/optimized/21816826_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                    'https://slimages.macysassets.com/is/image/MCY/products/9/optimized/22113049_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                    'https://slimages.macysassets.com/is/image/MCY/products/6/optimized/23818026_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                    ]
        althetic_list=[
                        "Men's Court Leather-Suede Sneaker",
                        "Men's Frisco Sneaker",
                        "Men's Run 70s Casual Sneakers from Finish Line",
                        "Men's Fly By Mid 3 Basketball Sneakers from Finish Line"
                    ]
        
        althetic_prices=[68.60,31.50,55.00,50.00]
        
        althetic_select = clickable_images(
            [
               'https://slimages.macysassets.com/is/image/MCY/products/7/optimized/21363567_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                'https://slimages.macysassets.com/is/image/MCY/products/6/optimized/21816826_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                'https://slimages.macysassets.com/is/image/MCY/products/9/optimized/22113049_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                'https://slimages.macysassets.com/is/image/MCY/products/6/optimized/23818026_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
            ],
            titles=althetic_list,
            div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
            img_style={"margin": "5px", "height": "200px"},
        )

        col1, col2, col3, col4 = st.columns([1,1,1,1])

        with(col1):
            st.write(althetic_list[0])
            st.write(f"${althetic_prices[0]}")
        with(col2):
            st.write(althetic_list[1])
            st.write(f"${althetic_prices[1]}")
        with(col3):
            st.write(althetic_list[2])
            st.write(f"${althetic_prices[2]}")
        with(col4):
            st.write(althetic_list[3])
            st.write(f"${althetic_prices[3]}")

        st.markdown(f"Scandal {althetic_list[althetic_select]} Selected" if althetic_select > -1 else "")

    elif(men_category_select==1):
        casual_urls=['https://slimages.macysassets.com/is/image/MCY/products/5/optimized/18038575_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                    'https://slimages.macysassets.com/is/image/MCY/products/5/optimized/19129775_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                    'https://slimages.macysassets.com/is/image/MCY/products/9/optimized/22240419_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                    'https://slimages.macysassets.com/is/image/MCY/products/0/optimized/23658370_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                    ]
        casual_list=[
                        "Men's Grand Series Jensen Sneakers",
                        "Men's Ryor Casual Slip-On Sneakers",
                        "Men's Faux-Leather Lace-Up Sneakers, Created for Macy's",
                        "Women's Ganimay Classic Ballet Flats"
                    ]
        
        casual_prices=[112.00,52.50,41.99,24.99]
        
        casual_select = clickable_images(
            [
                'https://slimages.macysassets.com/is/image/MCY/products/5/optimized/18038575_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                'https://slimages.macysassets.com/is/image/MCY/products/5/optimized/19129775_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                'https://slimages.macysassets.com/is/image/MCY/products/9/optimized/22240419_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                'https://slimages.macysassets.com/is/image/MCY/products/0/optimized/23658370_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
            ],
            titles=casual_list,
            div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
            img_style={"margin": "5px", "height": "200px"},
        )

        casual_col1, casual_col2, casual_col3, casual_col4 = st.columns([1,1,1,1])

        with(casual_col1):
            st.write(casual_list[0])
            st.write(f"${casual_prices[0]}")
        with(casual_col2):
            st.write(casual_list[1])
            st.write(f"${casual_prices[1]}")
        with(casual_col3):
            st.write(casual_list[2])
            st.write(f"${casual_prices[2]}")
        with(casual_col4):
            st.write(casual_list[3])
            st.write(f"${casual_prices[3]}")

        st.markdown(f"Flat {casual_list[casual_select]} Selected" if casual_select > -1 else "")
    elif(men_category_select==2): 
        # 2 - Dress Shoes
        dress_shoe_urls=['https://slimages.macysassets.com/is/image/MCY/products/3/optimized/9199563_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                    'https://slimages.macysassets.com/is/image/MCY/products/5/optimized/1461435_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                    'https://slimages.macysassets.com/is/image/MCY/products/1/optimized/8767971_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                    'https://slimages.macysassets.com/is/image/MCY/products/6/optimized/22281786_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                    ]
        dress_shoe_list=[
                        "Men's Andrew Plain Toe Derbys, Created for Macy's",
                        "Men's Brodie Lace Up Dress Oxford",
                        "Men's Tilden Walk Oxford",
                        "Men's Adeso Lace Up Dress Loafers"
                    ]
        
        dress_shoe_prices=[37.79,72.45,63.00,78.75]
        
        dress_shoe_select = clickable_images(
            [
                'https://slimages.macysassets.com/is/image/MCY/products/3/optimized/9199563_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                'https://slimages.macysassets.com/is/image/MCY/products/5/optimized/1461435_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                'https://slimages.macysassets.com/is/image/MCY/products/1/optimized/8767971_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                'https://slimages.macysassets.com/is/image/MCY/products/6/optimized/22281786_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
            ],
            titles=dress_shoe_list,
            div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
            img_style={"margin": "5px", "height": "200px"},
        )

        dress_shoe_col1, dress_shoe_col2, dress_shoe_col3, dress_shoe_col4 = st.columns([1,1,1,1])

        with(dress_shoe_col1):
            st.write(dress_shoe_list[0])
            st.write(f"${dress_shoe_prices[0]}")
        with(dress_shoe_col2):
            st.write(dress_shoe_list[1])
            st.write(f"${dress_shoe_prices[1]}")
        with(dress_shoe_col3):
            st.write(dress_shoe_list[2])
            st.write(f"${dress_shoe_prices[2]}")
        with(dress_shoe_col4):
            st.write(dress_shoe_list[3])
            st.write(f"${dress_shoe_prices[3]}")

        st.markdown(f"Scandal {dress_shoe_list[dress_shoe_select]} Selected" if dress_shoe_select > -1 else "")

    elif(men_category_select==3):
        # 3 - Loafers
        loafers_urls=['https://slimages.macysassets.com/is/image/MCY/products/6/optimized/23675776_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                    'https://slimages.macysassets.com/is/image/MCY/products/1/optimized/23537471_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                    'https://slimages.macysassets.com/is/image/MCY/products/4/optimized/15662224_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                    'https://slimages.macysassets.com/is/image/MCY/products/2/optimized/21019832_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                    ]
        loafers_list=[
                        "Women's Kinetic Impact II Lace-Up Mesh Sneakers",
                        "Women's Alondra Casual Platform Lace-up Sneakers",
                        "Women's Swift Run Casual Sneakers from Finish Line",
                        "Women's Chuck Taylor Madison Low Top Casual Sneakers from Finish Line"
                    ]
        
        loafers_prices=[145.00,63.30,70.00,60.00]
        
        loafers_select = clickable_images(
            [
                'https://slimages.macysassets.com/is/image/MCY/products/6/optimized/23675776_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                'https://slimages.macysassets.com/is/image/MCY/products/1/optimized/23537471_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                'https://slimages.macysassets.com/is/image/MCY/products/4/optimized/15662224_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
                'https://slimages.macysassets.com/is/image/MCY/products/2/optimized/21019832_fpx.tif?op_sharpen=1&wid=500&hei=611&fit=fit,1&fmt=webp',
            ],
            titles=loafers_list,
            div_style={"display": "flex", "justify-content": "center", "flex-wrap": "wrap"},
            img_style={"margin": "5px", "height": "200px"},
        )

        loafers_col1, loafers_col2, loafers_col3, loafers_col4 = st.columns([1,1,1,1])

        with(loafers_col1):
            st.write(loafers_list[0])
            st.write(f"${loafers_prices[0]}")
        with(loafers_col2):
            st.write(loafers_list[1])
            st.write(f"${loafers_prices[1]}")
        with(loafers_col3):
            st.write(loafers_list[2])
            st.write(f"${loafers_prices[2]}")
        with(loafers_col4):
            st.write(loafers_list[3])
            st.write(f"${loafers_prices[3]}")

        st.markdown(f"Sneaker {loafers_list[loafers_select]} Selected" if loafers_select > -1 else "")

with tab4:
    st.header("Recommendation System Analysis")

    st.write('### Recommendation Evaluation Performance')

    # Add intro text

    ###############
    from sklearn import datasets
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.pipeline import make_pipeline
    from sklearn.metrics import confusion_matrix
    bc = datasets.load_breast_cancer()
    X = bc.data
    y = bc.target

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1, stratify=y)
    pipeline = make_pipeline(StandardScaler(),
        RandomForestClassifier(n_estimators=10, max_features=5, max_depth=2, random_state=1))
    pipeline.fit(X_train, y_train)
    # Get the predictions
    y_pred = pipeline.predict(X_test)
    # Calculate the confusion matrix
    conf_matrix = confusion_matrix(y_true=y_test, y_pred=y_pred)

    st.write('### Classification Metrics')

    st.write('At Cornell Tech Boutique, we are transparent about the machine learning model performance of the recommendation systems used on our website. We have provided results including evaluation metrics: precision, recall, true positives, true negatives, false positives, and false negatives. We share the results on a test dataset.')

    true_neg, false_pos, false_neg, true_pos = conf_matrix.ravel()

    precision = true_pos/(true_pos+false_pos)
    recall = true_pos / (true_pos + false_neg)

    st.write("Precision: %s" % precision)
    st.write("Recall: %s" % recall)

    st.write('There are {} false positives'.format(false_pos))
    st.write('There are {} false negatives'.format(false_neg))
    st.write('There are {} true positives'.format(true_pos))
    st.write('There are {} true negatives'.format(true_neg))

    st.write('### Classification Confusion Matrix')

    # Add intro text

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.matshow(conf_matrix, cmap=plt.cm.Blues, alpha=0.3)
    for i in range(conf_matrix.shape[0]):
        for j in range(conf_matrix.shape[1]):
            ax.text(x=j, y=i,s=conf_matrix[i, j], va='center', ha='center', size='xx-large')
    
    plt.xlabel('Predictions', fontsize=18)
    plt.ylabel('Actuals', fontsize=18)
    plt.title('Confusion Matrix', fontsize=18)
    plt.show()
    st.pyplot(fig)

    st.write('### Classification Precision-Recall')

    # Add intro text

    precision_all = []
    recall_all = []
    threshold_values = np.linspace(0.5, 1, num=100)

    from sklearn.metrics import recall_score, precision_score

    probabilities = pipeline.predict_proba(X_test)
    
    for threshold in threshold_values:
        predictions = apply_threshold(probabilities, threshold)

        precision = precision_score(y_test, predictions)
        recall = recall_score(y_test, predictions)
        
        precision_all.append(precision)
        recall_all.append(recall)

    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    plt.rcParams['figure.figsize'] = 7, 5
    plt.locator_params(axis = 'x', nbins = 5)
    plt.plot(recall_all, precision_all, 'b', linewidth=4.0, color = '#B0017F')
    plt.title('Precision recall curve (all)')
    plt.ylabel('Precision')
    plt.xlabel('Recall')
    plt.rcParams.update({'font.size': 16})

    st.pyplot(fig)
