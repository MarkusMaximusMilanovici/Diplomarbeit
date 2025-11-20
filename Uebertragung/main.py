import time
from luma.led_matrix.device import max7219
from luma.core.interface.serial import spi, noop
from luma.core.render import canvas
from luma.core.legacy import text
from luma.core.legacy.font import proportional, LCD_FONT

serial = spi(port=0, device=0, gpio=noop())
device = max7219(serial, cascaded=4, block_orientation=-90)
device.contrast(10)

lst = """I might swerve, bend that corner, whoa
Bitch, hold on tight, 'cause I tweak in this bitch, start letting shit go
And I heard that she wanna show
Me who she be, I'm kinda fuckin' with it, show me some mo'
Bitch, we tatted head to toe, could give a fuck, the story wrote
You wanna tweak it up with me, then I'ma show you how that go
Like the money in my pockets, blow
They havin' convos about me, these pussy niggas don't know
Tell me what they talkin' 'bout, I ain't fuckin' listenin'
Let yo' thoughts run yo' mouth, but ain't touchin' dividends
Nigga, I ain't from the south, but kick it with my Memphis twin
Nigga, I can't take a loss, I'm always goin' for the win
I've been geeked up in this booth
I got my blunt packed from the start
Nigga, don't be there actin' new
That'll put some holes all through yo' body
I'm like, "Oh, them hoes is cool, let 'em in if that shit water"
See your nigga actin' bothered, got my green light to red dot 'em
If you spot me, ho, I'm sorry, I can't take you and get gnarly
Look like yo' nigga want smoke
Well, we gon' do this shit regardless
And I started from the bottom, just like you, but I was harder
I came up a fuckin' soldier, nigga
Shout out to my father (shout out to my father)
He made sure I'll make it farther
My trust in God, bucks, and dollars
I might swerve, bend that corner, whoa
Bitch, hold on tight, 'cause I tweak in this bitch, start lettin' shit go
And I heard that she wanna show
Me who she be, I'm kinda fuckin' with it, show me some mo'
Bitch, we tatted head to toe, could give a fuck, the story wrote
You wanna tweak it up with me, then I'ma show you how that go
Like the money in my pockets, blow
They havin' convos about me, these pussy niggas don't know
With my evil twin, all black hoodies, we hit the streets again
He don't like attention, he'll tweak and get to reapin' shit
What yo' ass expect? We drove the whole way in a demon, bitch
He'll pop out, won't think twice and make you greet the switch
Ah-ha-ha, is you sure you wanna meet that bitch?
If I see ill intentions, I'll make sure you get to bleedin' quick
All exotic whips, the whole damn gang be in some rocket shits
Pull up to the scene, my best advice is to hide yo' bitch
I could see the lies all on your face, because your eyes'll twitch
Black ski mask all on our fuckin' faces be disguisin' shit
I might have my way with yo' bay-bay, you wanna cry and shit
Rest my Glock against her fucking waist, I got some pottery
Ah-ah-ah, ah-ah-ah
She my bitch now, my apologies
I don't need no mo' enemies, you rock with me? Then rock with me
But I'ma keep it real, she lost respect when you said, "Follow me"
I might swerve, bend that corner, whoa
Bitch, hold on tight, 'cause I tweak in this bitch, start lettin' shit go
And I heard that she wanna show
Me who she be, I'm kinda fuckin' with it, show me some mo'
Bitch, we tatted head to toe, could give a fuck, the story wrote
You wanna tweak it up with me, then I'ma show you how that go
Like the money in my pockets, blow
They havin' convos about me, these pussy niggas don't know over
"""
index = 0

while True:
    with canvas(device) as draw:
        # Zentrierung: x=2,y=1 für jedes Modul (optimale Werte für LCD_FONT)
        text(draw, (2, 1), lst[(index+0)%len(lst)], fill="white", font=proportional(LCD_FONT))
        text(draw, (10, 1), lst[(index+1)%len(lst)], fill="white", font=proportional(LCD_FONT))
        text(draw, (18, 1), lst[(index+2)%len(lst)], fill="white", font=proportional(LCD_FONT))
        text(draw, (26, 1), lst[(index+3)%len(lst)], fill="white", font=proportional(LCD_FONT))
    time.sleep(0.1)
    index += 1
