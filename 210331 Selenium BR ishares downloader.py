from selenium import webdriver
from selenium.webdriver.common.keys import Keys
from selenium.webdriver.common.action_chains import ActionChains

#Set up driver
driver = webdriver.Chrome('./AppData/Local/Programs/Python/Python39/chromedriver')
driver.implicitly_wait(20)

#Open downloading website
driver.get("https://www.ishares.com/us/products/etf-investments#!type=ishares&style=All&view=keyFacts")
driver.maximize_window()

#Outer 'page' loop to scroll through 25 listings at a time
for page in range(16):
    scroll_y = 0
    #Inner link loop to scross through each of the ETFs

    for i in range(25):
        index = i + 1

        #for scrolling if link out of range (#Selenium needs the link or element to be in view)
        if index == 6 or index == 18:   #picked with trial and error
            scroll_y += 750  #picked with trial and error
            #print(f'Scroll Height: {driver.execute_script("return document.body.scrollHeight")}')  #for debugging
            driver.execute_script(f'window.scrollTo(0,{scroll_y});')    #actually scrolls
            _ = [x ** 2 for x in range(10000000)]  #to slow the program down and allow the website to l oad
            print('Scrolled Down!')    #for debugging

        element = driver.find_element_by_xpath(
            f'//*[@id="max-width-wrapper"]/ipx-table/ipx-desktop-table/div/table/tbody/tr[{index}]/td[2]/a[1]/span[2]') #ETF link

        print(f'{index}. Opening... {element.text}')    #for debugging

        #opens link in new window
        ActionChains(driver) \
                .move_to_element(element) \
                .key_down(Keys.CONTROL) \
                .click(element) \
                .key_up(Keys.CONTROL) \
                .perform()

        _ = [x ** 2 for x in range(10000000)]  #another forced pause

        print('Right-click complete!')  #for debugging

        #label tabs and then switch
        window_before = driver.window_handles[0]
        window_after = driver.window_handles[1]
        driver.switch_to.window(window_after)

        print(f'Opened! {driver.title}')  #for debugging

        #Find download element
        window_element = driver.find_element_by_link_text('Download')
        window_element.click()
        
        print('Waiting... ')
        _ = [x ** 2 for x in range(30000000)] #needs an even longer wait

        #close out tab and return back
        driver.close()
        driver.switch_to.window(window_before)

    #after list complete, find arrow forward
    next_page_element = driver.find_element_by_xpath('//*[@id="max-width-wrapper"]/ipx-table/div[3]/app-paginator/div/ul/li[9]')
    next_page_element.click()
    _ = [x ** 2 for x in range(10000000)]   #more forced waiting

    #scroll to top
    driver.execute_script('window.scrollTo(0,0);')
