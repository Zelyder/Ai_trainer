import pickle


def main():
    pkl_file = "ntu60_hrnet.pkl"

    with open(pkl_file, 'rb') as f:
        raw = pickle.load(f, encoding='latin1')

    sample = raw['annotations'][0]
    print("Тип элемента:", type(sample))
    print("Ключи:", sample.keys())
    print("Значения:")
    for k, v in sample.items():
        print(f"  {k}: {type(v)} -> {str(v)[:100]}")


if __name__ == "__main__":
    main()
