class FeatureEngineer:
    def __init__(self):
        self.full_bath = ['BsmtFullBath', 'FullBath']
        self.half_bath = ['BsmtHalfBath', 'HalfBath']
        self.tot_area = [
            'LotFrontage','LotArea','MasVnrArea','TotalBsmtSF','1stFlrSF','2ndFlrSF',
            'GarageArea','WoodDeckSF','OpenPorchSF','EnclosedPorch','3SsnPorch','ScreenPorch','PoolArea'
        ]

    def drop_columns(self, df, drop_list):
        return df.drop(drop_list, axis=1)

    def add_total_area(self, df):
        df['total_area'] = df[self.tot_area].sum(axis=1)
        df = df.drop(self.tot_area, axis=1)
        return df

    def add_total_bathrooms(self, df):
        df['totalbathreoams'] = df[self.full_bath].sum(axis=1) + (df[self.half_bath].sum(axis=1) * 0.5)
        df = df.drop(self.full_bath + self.half_bath, axis=1)
        return df