# -*- coding: utf-8 -*-
import string
import argparse

import warnings
warnings.filterwarnings('ignore')

import torch
import torch.backends.cudnn as cudnn
import torch.utils.data

from utils import CTCLabelConverter, AttnLabelConverter
from dataset import RawDataset, AlignCollate
from model import Model
import logging
import pandas as pd
from datetime import datetime


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y%m%d %H:%M:%S')


def demo(opt):

    file_names = []
    predictions = []

    """ model configuration """
    if 'CTC' in opt.Prediction:
        converter = CTCLabelConverter(opt.character)
    else:
        converter = AttnLabelConverter(opt.character)
    opt.num_class = len(converter.character)

    if opt.rgb:
        opt.input_channel = 3
    model = Model(opt)

    model = torch.nn.DataParallel(model)
    if torch.cuda.is_available():
        model = model.cuda()

    # load model
    logging.info('Loading pretrained model from %s' % opt.saved_model)
    model.load_state_dict(torch.load(opt.saved_model, map_location=torch.device('cpu')))

    # prepare data. two demo images from https://github.com/bgshih/crnn#run-demo
    AlignCollate_demo = AlignCollate(imgH=opt.imgH, imgW=opt.imgW, keep_ratio_with_pad=opt.PAD)
    demo_data = RawDataset(root=opt.image_folder, opt=opt)  # use RawDataset
    demo_loader = torch.utils.data.DataLoader(
        demo_data, batch_size=opt.batch_size,
        shuffle=False,
        num_workers=int(opt.workers),
        collate_fn=AlignCollate_demo, pin_memory=True)
    
    counter = 0
    # logging.info('-' * 80)
    # logging.info('menu_image_path\tpredicted_labels')
    # logging.info('-' * 80)
    # predict
    model.eval()
    for image_tensors, image_path_list in demo_loader:
        batch_size = image_tensors.size(0)
        with torch.no_grad():
            # image = image_tensors.cuda()
            # # For max length prediction
            # length_for_pred = torch.cuda.IntTensor([opt.batch_max_length] * batch_size)
            # text_for_pred = torch.cuda.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0)

            image = image_tensors
            length_for_pred = torch.IntTensor([opt.batch_max_length] * batch_size)
            text_for_pred = torch.LongTensor(batch_size, opt.batch_max_length + 1).fill_(0)


        if 'CTC' in opt.Prediction:
            preds = model(image, text_for_pred).log_softmax(2)

            # Select max probabilty (greedy decoding) then decode index to character
            preds_size = torch.IntTensor([preds.size(1)] * batch_size)
            _, preds_index = preds.permute(1, 0, 2).max(2)
            preds_index = preds_index.transpose(1, 0).contiguous().view(-1)
            preds_str = converter.decode(preds_index.data, preds_size.data)

        else:
            preds = model(image, text_for_pred, is_train=False)

            # select max probabilty (greedy decoding) then decode index to character
            _, preds_index = preds.max(2)
            preds_str = converter.decode(preds_index, length_for_pred)

        for img_name, pred in zip(image_path_list, preds_str):
            counter += 1
            
            if 'Attn' in opt.Prediction:
                pred = pred[:pred.find('[s]')]  # prune after "end of sentence" token ([s])

            file_names.append(img_name)
            predictions.append(pred)
            # logging.info(f'Step# {counter}: {img_name}\t{pred}')
            
            if counter % 5000 == 0:
                logging.info(f'Step# {counter}: {img_name}\t{pred}')
    
    return file_names, predictions


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--image_folder', required=True, help='path to image_folder which contains text images')
    parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
    parser.add_argument('--batch_size', type=int, default=192, help='input batch size')
    parser.add_argument('--saved_model', required=True, help="path to saved_model to evaluation")
    """ Data processing """
    parser.add_argument('--batch_max_length', type=int, default=50, help='maximum-label-length')
    # parser.add_argument('--imgH', type=int, default=32, help='the height of the input image')
    # parser.add_argument('--imgW', type=int, default=400, help='the width of the input image')
    parser.add_argument('--imgH', type=int, default=64, help='the height of the input image')
    parser.add_argument('--imgW', type=int, default=100, help='the width of the input image')

    parser.add_argument('--rgb', action='store_true', help='use rgb input')

    # parser.add_argument('--character', type=str, default='0123456789abcdefghijklmnopqrstuvwxyz', help='character label')
    parser.add_argument('--character', type=str,
                        default='0123456789abcdefghijklmnopqrstuvwxyzABDEFGHIJLMNQRT가각간갈감갑값갓강같걀개객거건걸검것겉게겨격견결겹경곁계고곡곤곧골곰곱공과관광괴교구국군굴굽궁권귀규균귤그극근글금급긋기긴길김까깍깐깔깜깡깨껌껍껏껑께꼬꼴꼼꼽꽂꽃꽉꽤꾸꿀꿈뀌끄끈끌끓끔끝끼낌나낙난날남납낫낭낮낱내냄냉냐냥너널넓넘네넷녀녁년념녕노녹논놀놈농놓뇨누눈뉴느늑는늘능늦늬니닐님다닥단달닭담답닷당대댁댓더덕던덜덤덥덧덩덮데델도독돈돌돗동돼되된두둑둘둠둥뒤뒷드득든듣들듬듯등디딩따딱딴딸땀땅때떠떡떼또똑뚜뚱뛰뜨뜰뜻라락란람랍랑래랜램랫량러럭런럴럼럽렁레렉렌려력련령례로록론롬롭롯료루룩룻뤄류륙률르른름릇릎리릭린림립릿링마막만많말맑맘맛망맞매맥맨맵머먹먼멀멍메멘멩면멸명모목몬몰몸못몽묘무묵묶문물뭇므미민밀및밑바박반받발밤밥방밭배백뱃버번벌범법베벤벨벽변별병보복볶본볼봄봇봉뵈부북분불붉붐붓붕뷰브븐블비빌빔빗빚빛빠빨빵빼뻐뻔뼈뿌쁘사삭산살삶삼삿상새색샌생샤서석섞선설섬섭섯성세센셈셋셔션소속손솔솜송솥쇠쇼수숙순술숨숫숭쉬슈스슨슬슴승시식신실심싱싶싸싹싼쌀쌍써썩썰쎄쏘쓰씨씩씬아악안알암압앗앙앞애액야약얀얄얇양어억언얼엄업없엇엉에엔엘여역연열염엽영예옛오옥온올옷옹와완왕외요욕용우욱운울움웃웅워원월웨위윗유육율으은을음응의이익인일임입잇있잎자작잔잘잠잡잣장재쟁저적전절점접젓정젖제젠젯져조족존졸종좋좌주죽준줄중쥐즈즉즌즐증지직진질짐집짓징짚짜짝째쪽쭈쭉찌찍찢차착찬찰참창채챙처척천철첩청체쳐초촉촌총최추춘출춧충취츠치친칠침칭카칸칼캄캐커컨컬컵컷케켓코콘콜콤콩쾌쿄쿠퀴크큰클큼키킬타탁탄탈탑탕태택터털텅테텍텔템토톤톨통퇴투툼튀튜트특튼틀틈티틱팀팅파팍판팔팝패팩팬퍼페펜편평폐포폭폰표푸풀품풍퓨프플피픽필핏핑하학한할함합항해핵핸햄햇행향허헌험헤헬혀현혈형혜호혹혼홀홈홉홍화환활황회횟횡효후훈훌훨휴흐흑흔흙흥희흰히힘?!:;@#$%^&*()[]-_=+~|₩<>.,/\'"三代一二元八古名品大中小人火牛眞生特本갱겐겔겟곶궈깃깹깻깽꼰꽁꽈꿍꿔낄낵낸넙넛넨넬넴넹놋뇽눌닉닌닛닝댈댕덴뎅돔될둔둣듀딕딜딤딥딧땡똠똥뚝뜸띠띳랄랏랙랩랭렐렘렛렝롤롱롼뢰룡룸룽뤼륜릴맷멕멜멤멧뭉믈믐믹밋밴밸뱅벅벡벳붇뷔빅빈빕빙빤뺀뺑뺘뻬뽀뽁뽄뽈뽕뿍뿔뿜뿡쁠삐삑삥삽샐샘샥샬샵샷샹셀셜셰셸숄숏숯쉐쉑쉘슉슐쌈쌩썬썸쎈쏨쏸쑝쑥앤앱앵얌엑엠엣옌옐옵옻욘욤웍웰윈윙잉잭잼잿젤쥰즙짠짬쪼쫀쫄쫑쯔찐찜찡찹챗챠챤챱첸촨촵쵸츄칡칩칵캉캑캔캘캡켄켈콕콥콰콸쿡쿤쿨쿰쿵쿼퀘퀸큐킨킷킹탐탠탭탱텃텐톡톰톳툭툰틴틸팃팜팟팡팥팸팻팽펀펄펍펠펩폴폼퐁푼풋픈핀핫헛헝헨훗훙훠훼휠힌힐앉멈꺼녑씀낚습릅를쥬갯첫푀죠찾힙확얹밝춤갖칙릉룹쌉굳흘씽큘쑤획굿辛쩡켜뱀렇잊죄꼭宴걱쏟깥춥딹빡겁캠뇌숲끗펑축덟꿩딪밖책곳벗툴珍펌괜턴빽튤십쟈킴믄豚칫높층羊럿엿맡춉',
                        help='character label')
    parser.add_argument('--sensitive', action='store_true', help='for sensitive character mode')
    parser.add_argument('--PAD', action='store_true', help='whether to keep ratio then pad for image resize')
    """ Model Architecture """
    parser.add_argument('--Transformation', type=str, required=True, help='Transformation stage. None|TPS')
    parser.add_argument('--FeatureExtraction', type=str, required=True, help='FeatureExtraction stage. VGG|RCNN|ResNet')
    parser.add_argument('--SequenceModeling', type=str, required=True, help='SequenceModeling stage. None|BiLSTM')
    parser.add_argument('--Prediction', type=str, required=True, help='Prediction stage. CTC|Attn')
    parser.add_argument('--num_fiducial', type=int, default=20, help='number of fiducial points of TPS-STN')
    parser.add_argument('--input_channel', type=int, default=1, help='the number of input channel of Feature extractor')
    parser.add_argument('--output_channel', type=int, default=512,
                        help='the number of output channel of Feature extractor')
    parser.add_argument('--hidden_size', type=int, default=256, help='the size of the LSTM hidden state')

    opt = parser.parse_args()

    """ vocab / character number configuration """
    if opt.sensitive:
        opt.character = string.printable[:-6]  # same with ASTER setting (use 94 char).

    cudnn.benchmark = True
    cudnn.deterministic = True
    opt.num_gpu = torch.cuda.device_count()
    print()
    logging.info(f'Start predictions....')
    img_path_list, pred_list = demo(opt)
    logging.info(f'Completed predictions....')

    print()
    logging.info(f'Creating result file....')
    df_prd = pd.DataFrame(zip(img_path_list, pred_list), columns=['menu_image_file', 'prediction'])
    
    gt_texts = [el.split('%')[0].strip().split('/')[-1] for el in img_path_list]

    df_prd['gt_text'] = gt_texts
    df_prd['prediction'] = df_prd['prediction'].fillna('')
    df_prd = df_prd[['menu_image_file', 'gt_text', 'prediction']]

    final_res_path = f'data/result/2-084_text-recognition_result_{datetime.today().strftime("%m%d%H%M%S")}.xlsx'
    try:
        df_prd.to_excel(final_res_path, index=False)
    except:
        df_prd.to_excel(final_res_path, index=False, engine='xlsxwriter')

    logging.info(f'[{final_res_path}] is created!')

    print(f'....Done!')
    